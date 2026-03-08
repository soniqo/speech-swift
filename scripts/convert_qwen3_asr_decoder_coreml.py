#!/usr/bin/env python3
"""
Convert Qwen3-ASR-0.6B text decoder to CoreML with MLState KV cache.

Produces two CoreML models:
  1. embedding.mlpackage — token_id → embedding lookup (stateless)
  2. decoder.mlpackage   — single-step decoder with KV cache states

The decoder uses ct.StateType for persistent KV cache on device (MLState),
requiring macOS 15+ / iOS 18+. Fixed-size cache with attention mask and
one-hot cache update pattern for ANE-friendly elementwise ops.

Architecture: 28-layer Qwen3 transformer with GQA (16Q/8KV heads),
split-half RoPE, SwiGLU MLP, per-head Q/K RMSNorm, tied LM head.

Requires:
  pip install torch coremltools safetensors numpy huggingface_hub

Usage:
  python scripts/convert_qwen3_asr_decoder_coreml.py
  python scripts/convert_qwen3_asr_decoder_coreml.py --output-dir ./qwen3-asr-decoder-coreml
  python scripts/convert_qwen3_asr_decoder_coreml.py --compile
"""

import argparse
import gc
import json
import math
import shutil
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Decoder config for Qwen3-ASR-0.6B ──

MAX_SEQ = 1024  # Fixed KV cache size

DECODER_CONFIG = {
    "hidden_size": 1024,
    "num_layers": 28,
    "num_heads": 16,        # query heads
    "num_kv_heads": 8,      # KV heads (GQA)
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
}


# ── PyTorch model components ──

class RMSNorm(nn.Module):
    """RMSNorm matching Qwen3 implementation."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        norm = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * norm * self.weight).to(x.dtype)


class SplitHalfRoPE(nn.Module):
    """Split-half rotary position embeddings (traditional=false in MLX).

    Computes rotation from position input for CoreML tracing compatibility.
    """

    def __init__(self, head_dim, theta=1000000.0):
        super().__init__()
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position):
        # x: [B, N_heads, 1, head_dim], position: [1] int32
        freqs = position.float() * self.inv_freq  # [half]
        cos_val = torch.cos(freqs).view(1, 1, 1, -1)  # [1, 1, 1, half]
        sin_val = torch.sin(freqs).view(1, 1, 1, -1)  # [1, 1, 1, half]

        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([
            x1 * cos_val - x2 * sin_val,
            x2 * cos_val + x1 * sin_val,
        ], dim=-1)


class DecoderAttention(nn.Module):
    """GQA attention with in-place KV cache buffer updates.

    Cache buffers are modified in-place via mul_() and add_() so that
    coremltools can map them to MLState operations.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.head_dim = config["head_dim"]
        self.groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        hidden = config["hidden_size"]
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.rope = SplitHalfRoPE(self.head_dim, theta=config["rope_theta"])

    def forward(self, x, position, mask, k_cache, v_cache, onehot):
        # x: [1, 1, hidden], position: [1], mask: [1, 1, 1, MAX_SEQ]
        # k_cache/v_cache: buffer refs [1, 8, MAX_SEQ, 128]
        # onehot: [1, 1, MAX_SEQ, 1] precomputed
        B, T, _ = x.shape  # T=1

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head Q/K normalization (Qwen3 specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE rotation
        q = self.rope(q, position)
        k = self.rope(k, position)

        # In-place cache update: zero out old position, insert new K/V
        # mul_(1 - onehot): clears position p (onehot=1 → multiply by 0)
        # add_(k * onehot): inserts new k at position p
        k_cache.mul_(1 - onehot)
        k_cache.add_(k * onehot)
        v_cache.mul_(1 - onehot)
        v_cache.add_(v * onehot)

        # GQA: expand KV heads to match Q heads
        k_exp = k_cache.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
        k_exp = k_exp.reshape(B, self.num_heads, MAX_SEQ, self.head_dim)
        v_exp = v_cache.unsqueeze(2).expand(-1, -1, self.groups, -1, -1)
        v_exp = v_exp.reshape(B, self.num_heads, MAX_SEQ, self.head_dim)

        # Scaled dot-product attention with mask
        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)

        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class DecoderMLP(nn.Module):
    """SwiGLU MLP matching Qwen3 architecture."""

    def __init__(self, config):
        super().__init__()
        hidden = config["hidden_size"]
        ffn = config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Single decoder layer with pre-norm, attention, and SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        self.self_attn = DecoderAttention(config)
        self.mlp = DecoderMLP(config)
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, x, position, mask, k_cache, v_cache, onehot):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position, mask, k_cache, v_cache, onehot)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen3ASRDecoder(nn.Module):
    """Full Qwen3-ASR text decoder with registered buffer KV cache.

    KV cache tensors are registered as named buffers so coremltools can
    map them to MLState via ct.StateType. The forward pass reads from
    buffers, computes updated cache, and writes back in-place via copy_().

    Forward signature: (input_embeds, position, mask) → logits
    """

    def __init__(self, config=None):
        super().__init__()
        c = config or DECODER_CONFIG
        self.num_layers = c["num_layers"]
        self.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c["num_layers"])])
        self.norm = RMSNorm(c["hidden_size"], eps=c["rms_norm_eps"])
        # LM head (tied with embedding — loaded from same weight)
        self.lm_head = nn.Linear(c["hidden_size"], c["vocab_size"], bias=False)

        # Register KV cache buffers (mapped to MLState by coremltools)
        num_kv = c["num_kv_heads"]
        hd = c["head_dim"]
        for i in range(c["num_layers"]):
            self.register_buffer(f"k_cache_{i}", torch.zeros(1, num_kv, MAX_SEQ, hd))
            self.register_buffer(f"v_cache_{i}", torch.zeros(1, num_kv, MAX_SEQ, hd))

    def forward(self, input_embeds, position, mask):
        # input_embeds: [1, 1, 1024], position: [1], mask: [1, 1, 1, MAX_SEQ]
        x = input_embeds

        # Precompute one-hot for cache updates (shared across all layers)
        onehot = F.one_hot(position.long(), MAX_SEQ).float().view(1, 1, MAX_SEQ, 1)

        for i in range(self.num_layers):
            k_buf = getattr(self, f"k_cache_{i}")
            v_buf = getattr(self, f"v_cache_{i}")
            # Layers update k_buf/v_buf in-place via mul_() + add_()
            x = self.layers[i](x, position, mask, k_buf, v_buf, onehot)

        x = self.norm(x)
        logits = self.lm_head(x)  # [1, 1, vocab_size]
        return logits


class EmbeddingLookup(nn.Module):
    """Token embedding lookup for CoreML conversion."""

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, token_id):
        # token_id: [1, 1] int32
        return self.embedding(token_id)  # [1, 1, hidden_size]


# ── Weight loading ──

def download_decoder_weights(model_id, weights_dir=None):
    """Load text decoder weights from HuggingFace.

    Returns dict with keys like:
        layers.0.self_attn.q_proj.weight
        layers.0.mlp.gate_proj.weight
        embed_tokens.weight
        norm.weight
    """
    from safetensors.torch import load_file

    if weights_dir:
        cache_dir = Path(weights_dir)
        print(f"Loading weights from {cache_dir}...")
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading weights from {model_id}...")
        cache_dir = Path(snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        ))

    all_weights = {}
    for f in sorted(cache_dir.glob("*.safetensors")):
        print(f"  Loading {f.name}...")
        all_weights.update(load_file(str(f)))

    # Strip prefix: thinker.model.X or model.X
    text_weights = {}
    for k, v in all_weights.items():
        for prefix in ["thinker.model.", "model."]:
            if k.startswith(prefix):
                text_weights[k[len(prefix):]] = v
                break

    print(f"  Extracted {len(text_weights)} text decoder weights")

    # Also look for embed_tokens directly if not found via prefix stripping
    if "embed_tokens.weight" not in text_weights:
        for k, v in all_weights.items():
            if k.endswith("embed_tokens.weight"):
                text_weights["embed_tokens.weight"] = v
                print(f"  Found embed_tokens via: {k}")
                break

    return text_weights


def load_decoder_weights(decoder, embedding_model, weights):
    """Load weights into PyTorch decoder and embedding models."""
    # Map HF keys → decoder model keys
    decoder_sd = decoder.state_dict()
    loaded = 0
    total = 0

    for key in decoder_sd:
        total += 1
        src_key = key

        # LM head uses tied embedding weight
        if key == "lm_head.weight":
            src_key = "embed_tokens.weight"

        # Skip buffers (KV cache, RoPE inv_freq)
        if "k_cache_" in key or "v_cache_" in key or "inv_freq" in key:
            continue

        if src_key in weights:
            if decoder_sd[key].shape == weights[src_key].shape:
                decoder_sd[key] = weights[src_key].float()
                loaded += 1
            else:
                print(f"  Shape mismatch: {key} model={decoder_sd[key].shape} weight={weights[src_key].shape}")
        else:
            print(f"  Missing: {key} (looked for: {src_key})")

    decoder.load_state_dict(decoder_sd)
    print(f"  Decoder: loaded {loaded}/{total} weights")

    # Load embedding weights
    emb_sd = embedding_model.state_dict()
    if "embed_tokens.weight" in weights:
        emb_sd["embedding.weight"] = weights["embed_tokens.weight"].float()
        embedding_model.load_state_dict(emb_sd)
        print(f"  Embedding: loaded 1/1 weights")
    else:
        print(f"  WARNING: embed_tokens.weight not found!")


# ── CoreML conversion ──

def convert_embedding(traced_embedding, quantize_nbits=None):
    """Convert embedding lookup to CoreML."""
    print("Converting embedding model...")
    mlmodel = ct.convert(
        traced_embedding,
        inputs=[ct.TensorType(name="token_id", shape=(1, 1), dtype=np.int32)],
        outputs=[ct.TensorType(name="embedding")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if quantize_nbits:
        print(f"  Applying INT{quantize_nbits} palettization...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig, OptimizationConfig, palettize_weights,
        )
        op_config = OpPalettizerConfig(mode="kmeans", nbits=quantize_nbits)
        config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, config)

    return mlmodel


def convert_decoder(traced_decoder, config, quantize_nbits=None):
    """Convert decoder to CoreML with MLState KV cache.

    Buffer names (k_cache_0, v_cache_0, ...) are mapped to ct.StateType
    so coremltools creates MLState entries in the CoreML model.
    """
    num_layers = config["num_layers"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]

    inputs = [
        ct.TensorType(name="input_embeds", shape=(1, 1, config["hidden_size"]), dtype=np.float32),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, 1, 1, MAX_SEQ), dtype=np.float32),
    ]

    # States map to registered buffers by name
    states = []
    for i in range(num_layers):
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, num_kv_heads, MAX_SEQ, head_dim)),
            name=f"k_cache_{i}",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, num_kv_heads, MAX_SEQ, head_dim)),
            name=f"v_cache_{i}",
        ))

    outputs = [ct.TensorType(name="logits")]

    print(f"Converting decoder ({num_layers}L, {len(states)} cache states)...")
    mlmodel = ct.convert(
        traced_decoder,
        inputs=inputs,
        states=states,
        outputs=outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if quantize_nbits:
        print(f"  Applying INT{quantize_nbits} palettization...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig, OptimizationConfig, palettize_weights,
        )
        op_config = OpPalettizerConfig(mode="kmeans", nbits=quantize_nbits)
        opt_config = OptimizationConfig(global_config=op_config)
        mlmodel = palettize_weights(mlmodel, opt_config)

    return mlmodel


def compile_mlpackage(output_dir, name):
    """Compile .mlpackage → .mlmodelc using xcrun."""
    pkg = output_dir / f"{name}.mlpackage"
    compiled = output_dir / f"{name}.mlmodelc"

    if compiled.exists():
        shutil.rmtree(compiled)

    print(f"Compiling {name}.mlpackage → {name}.mlmodelc...")
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(pkg), str(output_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  xcrun failed: {result.stderr.strip()}")
        print("  Falling back to Python compilation...")
        compiled_url = ct.utils.compile_model(str(pkg))
        shutil.move(str(compiled_url), str(compiled))

    if compiled.exists():
        print(f"  Compiled to {compiled}")
    else:
        print(f"  ERROR: {compiled} not found after compilation")


def verify_embedding(pt_model, coreml_path, n=5):
    """Verify CoreML embedding matches PyTorch."""
    print(f"\nVerifying embedding model ({n} tokens)...")
    coreml_model = ct.models.MLModel(str(coreml_path))
    max_diff = 0.0

    for token in [0, 1, 100, 1000, 151645]:
        tok = torch.tensor([[token]], dtype=torch.int32)
        with torch.no_grad():
            pt_out = pt_model(tok).numpy().flatten().astype(np.float32)

        cm_out = np.array(
            coreml_model.predict({"token_id": tok.numpy()})["embedding"]
        ).flatten().astype(np.float32)

        cos = float(np.dot(pt_out, cm_out) / (np.linalg.norm(pt_out) * np.linalg.norm(cm_out) + 1e-10))
        diff = 1.0 - cos
        max_diff = max(max_diff, diff)
        print(f"  token={token:6d}: cosine_sim={cos:.6f}")

    status = "PASS" if max_diff < 0.01 else "WARNING"
    print(f"  {status}: max (1-cosine_sim) = {max_diff:.6f}")
    return max_diff


def verify_decoder(pt_decoder, pt_embedding, coreml_decoder_path, coreml_embedding_path):
    """Run a few decoder steps and verify against PyTorch."""
    print("\nVerifying decoder (5-step autoregressive)...")

    cml_decoder = ct.models.MLModel(str(coreml_decoder_path))
    cml_embedding = ct.models.MLModel(str(coreml_embedding_path))

    # Create MLState for stateful prediction (macOS 15+ / iOS 18+)
    decoder_state = cml_decoder.make_state()

    config = DECODER_CONFIG
    num_layers = config["num_layers"]

    # Reset PyTorch cache buffers
    for i in range(num_layers):
        getattr(pt_decoder, f"k_cache_{i}").zero_()
        getattr(pt_decoder, f"v_cache_{i}").zero_()

    tokens = [151644, 8948, 198, 151645, 198]  # <|im_start|>system\n<|im_end|>\n
    match_count = 0

    for step, token in enumerate(tokens):
        # Build mask: allow positions 0..step
        mask = torch.full((1, 1, 1, MAX_SEQ), -1e4)
        mask[:, :, :, :step + 1] = 0
        mask_np = mask.numpy()
        position = torch.tensor([step], dtype=torch.int32)

        # PyTorch path (cache buffers updated in-place by forward)
        with torch.no_grad():
            tok_t = torch.tensor([[token]], dtype=torch.int32)
            embed = pt_embedding(tok_t)
            pt_logits_t = pt_decoder(embed, position, mask)
            pt_logits = pt_logits_t.detach().numpy().flatten()
            pt_token = int(np.argmax(pt_logits))

        # CoreML path (state managed by MLState)
        cm_embed = cml_embedding.predict({"token_id": np.array([[token]], dtype=np.int32)})
        cm_input = {
            "input_embeds": cm_embed["embedding"].astype(np.float32),
            "position": np.array([step], dtype=np.int32),
            "attention_mask": mask_np,
        }
        cm_result = cml_decoder.predict(cm_input, state=decoder_state)
        cm_logits = np.array(cm_result["logits"]).flatten().astype(np.float32)
        cm_token = int(np.argmax(cm_logits))

        cos = float(np.dot(pt_logits, cm_logits) / (
            np.linalg.norm(pt_logits) * np.linalg.norm(cm_logits) + 1e-10))
        match = "✓" if pt_token == cm_token else "✗"
        if pt_token == cm_token:
            match_count += 1
        print(f"  step={step}: cos_sim={cos:.6f}, pt_token={pt_token}, cm_token={cm_token} {match}")

    print(f"  Token match: {match_count}/{len(tokens)}")


# ── Main ──

def main():
    global MAX_SEQ

    parser = argparse.ArgumentParser(description="Convert Qwen3-ASR decoder to CoreML")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B",
                        help="HuggingFace model ID (default: Qwen/Qwen3-ASR-0.6B)")
    parser.add_argument("--weights-dir",
                        help="Local directory with safetensors (skip HF download)")
    parser.add_argument("--output-dir", default="./qwen3-asr-decoder-coreml",
                        help="Output directory")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip INT8 palettization")
    parser.add_argument("--compile", action="store_true",
                        help="Compile .mlpackage to .mlmodelc")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip verification step")
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ,
                        help=f"Max sequence length for KV cache (default: {MAX_SEQ})")
    args = parser.parse_args()

    MAX_SEQ = args.max_seq

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantize_nbits = None if args.no_quantize else 8

    config = DECODER_CONFIG

    # ── Phase 1: Download weights ──
    print("=" * 60)
    print("Phase 1: Download text decoder weights")
    print("=" * 60)
    weights = download_decoder_weights(args.model_id, args.weights_dir)

    # ── Phase 2: Build PyTorch models ──
    print("\n" + "=" * 60)
    print("Phase 2: Build PyTorch models")
    print("=" * 60)

    decoder = Qwen3ASRDecoder(config)
    embedding = EmbeddingLookup(config["vocab_size"], config["hidden_size"])

    dec_params = sum(p.numel() for p in decoder.parameters())
    emb_params = sum(p.numel() for p in embedding.parameters())
    print(f"  Decoder: {dec_params:,} params ({dec_params * 4 / 1024 / 1024:.1f} MB FP32)")
    print(f"  Embedding: {emb_params:,} params ({emb_params * 4 / 1024 / 1024:.1f} MB FP32)")

    # ── Phase 3: Load weights ──
    print("\n" + "=" * 60)
    print("Phase 3: Load weights")
    print("=" * 60)
    load_decoder_weights(decoder, embedding, weights)
    decoder.eval()
    embedding.eval()
    del weights
    gc.collect()

    # ── Phase 4: Trace models ──
    print("\n" + "=" * 60)
    print("Phase 4: Trace models")
    print("=" * 60)

    # Trace embedding
    example_token = torch.tensor([[5]], dtype=torch.int32)
    with torch.no_grad():
        traced_embedding = torch.jit.trace(embedding, (example_token,))

    # Verify trace
    with torch.no_grad():
        ref = embedding(example_token)
        trc = traced_embedding(example_token)
    diff = (ref - trc).abs().max().item()
    print(f"  Embedding trace diff: {diff:.2e}")
    assert diff < 1e-5, f"Embedding trace mismatch: {diff}"

    # Trace decoder (cache buffers are internal state, not function args)
    hidden = config["hidden_size"]

    example_embeds = torch.randn(1, 1, hidden)
    example_position = torch.tensor([0], dtype=torch.int32)
    example_mask = torch.zeros(1, 1, 1, MAX_SEQ)

    print(f"  Tracing decoder ({config['num_layers']} layers, {config['num_layers'] * 2} cache buffers)...")
    with torch.no_grad():
        traced_decoder = torch.jit.trace(
            decoder,
            (example_embeds, example_position, example_mask),
        )

    # Reset buffers after trace (trace modifies them in-place)
    for i in range(config["num_layers"]):
        getattr(decoder, f"k_cache_{i}").zero_()
        getattr(decoder, f"v_cache_{i}").zero_()

    # Verify trace
    with torch.no_grad():
        ref = decoder(example_embeds, example_position, example_mask)
        # Reset again
        for i in range(config["num_layers"]):
            getattr(decoder, f"k_cache_{i}").zero_()
            getattr(decoder, f"v_cache_{i}").zero_()
        trc = traced_decoder(example_embeds, example_position, example_mask)
    diff = (ref - trc).abs().max().item()
    print(f"  Decoder trace diff: {diff:.2e}")
    assert diff < 1e-5, f"Decoder trace mismatch: {diff}"

    # ── Phase 5: Convert to CoreML ──
    print("\n" + "=" * 60)
    print("Phase 5: Convert to CoreML")
    print("=" * 60)

    # Convert embedding
    mlmodel_emb = convert_embedding(traced_embedding, quantize_nbits=quantize_nbits)
    emb_path = output_dir / "embedding.mlpackage"
    if emb_path.exists():
        shutil.rmtree(emb_path)
    mlmodel_emb.save(str(emb_path))
    emb_size = sum(f.stat().st_size for f in emb_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  Saved embedding.mlpackage ({emb_size:.1f} MB)")
    del mlmodel_emb
    gc.collect()

    # Convert decoder
    mlmodel_dec = convert_decoder(traced_decoder, config, quantize_nbits=quantize_nbits)
    dec_path = output_dir / "decoder.mlpackage"
    if dec_path.exists():
        shutil.rmtree(dec_path)
    mlmodel_dec.save(str(dec_path))
    dec_size = sum(f.stat().st_size for f in dec_path.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  Saved decoder.mlpackage ({dec_size:.1f} MB)")
    del mlmodel_dec, traced_embedding, traced_decoder
    gc.collect()

    # ── Phase 6: Verify ──
    if not args.skip_verify:
        print("\n" + "=" * 60)
        print("Phase 6: Verify")
        print("=" * 60)
        verify_embedding(embedding, emb_path)
        verify_decoder(decoder, embedding, dec_path, emb_path)

    # ── Phase 7: Compile ──
    if args.compile:
        print("\n" + "=" * 60)
        print("Phase 7: Compile")
        print("=" * 60)
        compile_mlpackage(output_dir, "embedding")
        if (output_dir / "embedding.mlpackage").exists():
            shutil.rmtree(output_dir / "embedding.mlpackage")
        compile_mlpackage(output_dir, "decoder")
        if (output_dir / "decoder.mlpackage").exists():
            shutil.rmtree(output_dir / "decoder.mlpackage")

    del decoder, embedding
    gc.collect()

    # ── Save config ──
    out_config = {
        "model_type": "qwen3-asr-decoder-coreml",
        "source_model": args.model_id,
        "max_seq_length": MAX_SEQ,
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "num_kv_heads": config["num_kv_heads"],
        "head_dim": config["head_dim"],
        "intermediate_size": config["intermediate_size"],
        "vocab_size": config["vocab_size"],
        "rms_norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "quantization": f"int{quantize_nbits}_palettize" if quantize_nbits else "float16",
        "files": {
            "embedding": "embedding.mlpackage",
            "decoder": "decoder.mlpackage",
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(out_config, f, indent=2)
    print(f"\nSaved config.json")

    # Summary
    print(f"\nDone! Output in: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        sz = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file()) if f.is_dir() else f.stat().st_size
        print(f"  {f.name}: {sz / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
