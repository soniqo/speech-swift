"""Core ML export modules for Inkling's shared decoder layers.

Expert matrices stay INT8 until routing selects their rows. Convolution
histories and routing arithmetic retain FP32 precision.
"""
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted from the Apache-2.0 Inkling 2 reference equations. Modifications:
# PyTorch/Core ML partitions, INT8 storage, and explicit per-visit caches.
# See ThirdPartyNotices/Inkling2Mini.txt.
from __future__ import annotations

import math
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def quantize_int8(values: torch.Tensor, group_size: int = 64):
    """Symmetric quantization along input channels, including constant groups."""
    if values.ndim < 2 or group_size < 1 or values.shape[-1] % group_size:
        raise ValueError("INT8 matrices require complete input-channel groups")
    values = values.detach().float().contiguous()
    if not torch.isfinite(values).all():
        raise ValueError("Cannot quantize non-finite weights")
    groups = values.reshape(*values.shape[:-1], -1, group_size)
    scales = groups.abs().amax(-1).div(127).clamp_min(2 ** -24).half()
    if not torch.isfinite(scales).all():
        raise ValueError("Weight scales exceed the supported FP16 range")
    quantized = torch.round(groups / scales.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return quantized.reshape(values.shape), scales


class Int8Experts(nn.Module):
    """Gather INT8 experts before dequantizing the selected matrices."""

    def __init__(self, weight: torch.Tensor, group_size: int = 64,
                 compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        if weight.ndim != 3:
            raise ValueError("Experts require [experts, output, input] weights")
        quantized, scales = quantize_int8(weight, group_size)
        self.register_buffer("quantized", quantized)
        self.register_buffer("scales", scales)
        self.rows, self.columns = weight.shape[-2:]
        self.group_size = group_size
        self.compute_dtype = compute_dtype

    def forward(self, values, indices):
        # Keeping gather before cast prevents materializing every expert as FP16.
        selected = torch.index_select(self.quantized, 0, indices.reshape(-1))
        scales = torch.index_select(self.scales, 0, indices.reshape(-1))
        # INT8 values are exact in FP16, so FP16 multiply gives the same rounded
        # weights without a full FP32 selected-weight workspace.
        weights = (selected.reshape(-1, self.rows, self.columns // self.group_size,
                                    self.group_size).to(self.compute_dtype)
                   * scales.to(self.compute_dtype).unsqueeze(-1))
        weights = weights.reshape(-1, self.rows, self.columns)
        inputs = values.to(self.compute_dtype).expand(*indices.shape, self.columns).reshape(-1, self.columns, 1)
        return torch.bmm(weights, inputs).reshape(*indices.shape, self.rows)


class CompressedLinear(nn.Module):
    """Dense INT8 storage through Core ML's compression metadata interface."""

    def __init__(self, weight: torch.Tensor, group_size: int = 64,
                 compute_dtype: torch.dtype = torch.float16):
        super().__init__()
        quantized, scales = quantize_int8(weight, group_size)
        reconstructed = (quantized.reshape(*weight.shape[:-1], -1, group_size).float()
                         * scales.float().unsqueeze(-1)).reshape(weight.shape).to(compute_dtype)
        self.register_buffer("weight", reconstructed)
        self.register_buffer("_COREML_/metadata_version", torch.tensor(2))
        self.register_buffer("_COREML_/weight/compression_type", torch.tensor([3]))
        self.register_buffer("_COREML_/weight/quantization_n_bits", torch.tensor(8))
        self.register_buffer("_COREML_/weight/quantization_scale", scales.to(compute_dtype))
        self.compute_dtype = compute_dtype

    def forward(self, values):
        return F.linear(values.to(self.compute_dtype), self.weight)


def rms_norm(value, weight=None, epsilon=1e-6):
    value = value.float()
    normalized = value * torch.rsqrt((value * value).mean(-1, keepdim=True) + epsilon)
    return normalized if weight is None else normalized * weight.float()


def clipped(value, threshold, scale=1.0):
    scaled = value.float() * scale
    ratio = threshold / (torch.sqrt((scaled * scaled).sum(-1, keepdim=True)) + 1e-6)
    return (scaled * torch.minimum(ratio, torch.ones_like(ratio))).to(value.dtype)


def residual_mix(blocks, partial, query, normalize_values=True):
    states = torch.cat((blocks.float(), partial.float().unsqueeze(-2)), dim=-2)
    keys = rms_norm(states)
    scores = torch.matmul(keys, query.reshape(-1, partial.shape[-1]).T) / math.sqrt(partial.shape[-1])
    probabilities = torch.softmax(scores, dim=-2).transpose(-1, -2)
    return torch.matmul(probabilities, keys if normalize_values else states)


class Parameters(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.names = {}
        for index, (name, value) in enumerate(sorted(weights.items())):
            key = f"p{index}"
            self.register_buffer(key, torch.as_tensor(value).float())
            self.names[name] = key

    def get(self, name):
        return getattr(self, self.names[name])


class PlainLinear(nn.Module):
    def __init__(self, weight, dtype, bias=None):
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight).to(dtype))
        self.register_buffer("bias", None if bias is None else torch.as_tensor(bias).to(dtype))

    def forward(self, value):
        return F.linear(value.to(self.weight.dtype), self.weight, self.bias)


class PlainExperts(nn.Module):
    def __init__(self, weight, dtype):
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight).to(dtype))

    def forward(self, value, indices):
        selected = torch.index_select(self.weight, 0, indices.reshape(-1))
        rows, columns = self.weight.shape[-2:]
        inputs = value.to(self.weight.dtype).expand(*indices.shape, columns).reshape(-1, columns, 1)
        return torch.bmm(selected, inputs).reshape(*indices.shape, rows)


class Embedding(nn.Module):
    def __init__(self, config, weights, *, quantized=True, dtype=torch.float16, group_size=64):
        super().__init__()
        self.epsilon = config["rms_norm_eps"]
        self.dtype, self.quantized, self.group_size = dtype, quantized, group_size
        weight = torch.as_tensor(weights["language_model.model.embed_tokens.weight"])
        self.width = weight.shape[-1]
        # Core ML's embedding translation requires floating-point table values;
        # native compression metadata keeps the serialized table at INT8.
        self.table = (CompressedLinear(weight, group_size, dtype) if quantized
                      else PlainLinear(weight, dtype))
        self.register_buffer("norm", torch.as_tensor(weights["language_model.model.embed_norm.weight"]).float())

    def forward(self, input_ids):
        embedded = F.embedding(input_ids, self.table.weight)
        return rms_norm(embedded.to(self.dtype), self.norm, self.epsilon).to(self.dtype)


class DecoderLayer(nn.Module):
    """One physical layer; callers provide independent state for each visit."""

    def __init__(self, config, layer, weights, *, quantized=True, dtype=torch.float16, group_size=64):
        super().__init__()
        self.config, self.layer, self.dtype = config, layer, dtype
        prefix = f"language_model.model.layers.{layer}."
        message_queries = weights.get("language_model.model.message_queries")
        weights = {name.removeprefix(prefix): value for name, value in weights.items() if name.startswith(prefix)}
        self.local = layer in config["local_layer_ids"]
        loop = config["loop_config"]
        self.core = loop["core_start"] <= layer < loop["core_end"]
        self.linears, self.linear_names = nn.ModuleDict(), {}
        for name in list(weights):
            if name.startswith("self_attn.") and name.endswith(".weight") and ".conv." not in name:
                key = f"linear{len(self.linears)}"
                matrix = torch.as_tensor(weights.pop(name))
                # Dynamic convolution projections retain their source weights.
                compress = quantized and ".weight_proj." not in name
                target_dtype = torch.float32 if ".weight_proj." in name else dtype
                self.linears[key] = (CompressedLinear(matrix, group_size, dtype) if compress
                                     else PlainLinear(matrix, target_dtype))
                self.linear_names[name.removesuffix(".weight")] = key
        for projection in ["gate", "up", "down"]:
            name = f"mlp.switch_mlp.{projection}_proj.weight"
            matrix = torch.as_tensor(weights.pop(name))
            module = Int8Experts(matrix, group_size, dtype) if quantized else PlainExperts(matrix, dtype)
            self.add_module(projection, module)
        if self.core:
            if message_queries is None:
                raise ValueError("Core layers require the shared message queries")
            weights["message_query"] = torch.as_tensor(message_queries)[layer - loop["core_start"]]
        self.parameters_bank = Parameters(weights)

    @classmethod
    def from_weights(cls, config, layer, weights, **kwargs):
        return cls(config, layer, weights, **kwargs)

    def weight(self, name):
        return self.parameters_bank.get(name)

    def linear(self, value, name):
        return self.linears[self.linear_names[name]](value)

    def norm(self, value, name):
        return rms_norm(value, self.weight(name + ".weight"), self.config["rms_norm_eps"]).to(self.dtype)

    def controller(self, value, message, name, enabled):
        epsilon = self.config["rms_norm_eps"]
        pair = torch.cat((rms_norm(value, epsilon=epsilon), rms_norm(message, epsilon=epsilon)), dim=-1)
        hidden = F.linear(pair.to(self.dtype), self.weight(name + ".down_weight").to(self.dtype)).float()
        forget = torch.tanh(F.linear(hidden, self.weight(name + ".forget_weight")))
        gain = torch.tanh(F.linear(hidden, self.weight(name + ".gain_weight")))
        return value.float() + enabled * (-forget * value.float() + gain * message.float())

    def convolution(self, value, history, name, dynamic=None, heads=0):
        kernel = self.config["sconv_kernel_size"]
        context = torch.cat((history, value.float()), dim=1)
        if dynamic is None:
            weight = self.weight(name + ".conv.weight").permute(0, 2, 1)
            output = F.conv1d(context.transpose(1, 2), weight, groups=value.shape[-1]).transpose(1, 2)
            output = (output + value.float()) / math.sqrt(kernel + 1)
        else:
            taps = self.weight(name + ".static_weight").T.reshape(heads, -1, kernel)
            projected = self.linear(dynamic, name + ".weight_proj").float()
            taps = taps + projected.reshape(1, value.shape[1], heads, 1, kernel)
            output = torch.zeros_like(value.float()).reshape(1, value.shape[1], heads, -1)
            for lag in range(kernel):
                values = context[:, kernel - 1 - lag:kernel - 1 - lag + value.shape[1]]
                output = output + values.reshape(1, value.shape[1], heads, -1) * taps[..., lag]
            output = (output.reshape_as(value) + value.float()) / math.sqrt(2 * kernel + 1)
        return output.to(self.dtype), context[:, -(kernel - 1):]

    def attention(self, query_input, key_input, value_input, keys, values, valid, history_k, history_v, position):
        c = self.config
        heads = c["swa_num_attention_heads"] if self.local else c["num_attention_heads"]
        kv_heads = c["swa_num_key_value_heads"] if self.local else c["num_key_value_heads"]
        width = c["swa_head_dim"] if self.local else c["head_dim"]
        extent = c["sliding_window_size"] if self.local else c["rel_extent"]
        tokens = query_input.shape[1]
        prefix = "self_attn."
        key, history_k = self.convolution(self.linear(key_input, prefix + "k_proj"), history_k,
            prefix + "k_sconv", None if self.local else key_input, kv_heads)
        value, history_v = self.convolution(self.linear(value_input, prefix + "v_proj"), history_v,
            prefix + "v_sconv", None if self.local else value_input, kv_heads)
        query = clipped(self.linear(query_input, prefix + "q_proj").reshape(1, tokens, heads, width),
                        c["qk_l2_clip_threshold"], self.weight(prefix + "q_scale"))
        key = clipped(key.reshape(1, tokens, kv_heads, width), c["qk_l2_clip_threshold"])
        value = value.reshape(1, tokens, kv_heads, width)
        if c["use_value_norm_no_gamma"]:
            value = rms_norm(value, epsilon=c["rms_norm_eps"]).to(self.dtype)
        relative = clipped(self.linear(query_input, prefix + "r_proj").reshape(1, tokens, heads, c["d_rel"]),
                           c["rel_l2_clip_threshold"], self.weight(prefix + "rel_scale"))
        if not self.local and c["log_scaling_n_floor"] > 0:
            positions = position.float() + torch.arange(query_input.shape[1]).float() + 1
            tau = 1 + c["log_scaling_alpha"] * torch.log(torch.clamp(positions / c["log_scaling_n_floor"], min=1))
            query = (query.float() * tau.reshape(1, -1, 1, 1)).to(self.dtype)
            relative = (relative.float() * tau.reshape(1, -1, 1, 1)).to(self.dtype)
        query = query.permute(0, 2, 1, 3)
        keys = torch.cat((keys, key.permute(0, 2, 1, 3)), dim=2)
        values = torch.cat((values, value.permute(0, 2, 1, 3)), dim=2)
        distances = (torch.arange(query_input.shape[1]) + valid.shape[0]).unsqueeze(-1) - torch.arange(keys.shape[2])
        # A 2-D linear projection avoids FP16 batched-matmul shape expansion in
        # Core ML while preserving the learned relative-position equation.
        logits = F.linear(relative.reshape(-1, c["d_rel"]),
                          self.weight(prefix + "rel_proj").T.to(self.dtype))
        logits = logits.reshape(1, query_input.shape[1], heads, extent).permute(0, 2, 1, 3)
        indices = distances.clamp(0, extent - 1).reshape(1, 1, query_input.shape[1], -1).expand(1, heads, -1, -1)
        mask = torch.gather(logits, -1, indices)
        mask = torch.where(distances >= extent, torch.zeros_like(mask), mask)
        allowed = distances >= 0
        if self.local:
            allowed = allowed & (distances < c["sliding_window_size"])
        valid = torch.cat((valid, torch.ones(query_input.shape[1])))
        allowed = allowed & (valid > 0)
        mask = torch.where(allowed, mask, torch.full_like(mask, -float("inf")))
        sink = self.linear(query_input, prefix + "sink_proj").permute(0, 2, 1).unsqueeze(-1)
        mask = torch.cat((mask, sink), dim=-1)
        padded_k = torch.cat((keys, torch.zeros_like(keys[:, :, :1])), dim=2)
        padded_v = torch.cat((values, torch.zeros_like(values[:, :, :1])), dim=2)
        padded_k = padded_k.repeat_interleave(heads // kv_heads, dim=1)
        padded_v = padded_v.repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.matmul(query.float(), padded_k.float().transpose(-1, -2)) / math.sqrt(width) + mask.float()
        output = torch.matmul(torch.softmax(scores, dim=-1), padded_v.float()).to(self.dtype)
        output = output.permute(0, 2, 1, 3).reshape(1, tokens, heads * width)
        output = (output.float() * torch.sigmoid(self.linear(query_input, prefix + "gate_proj").float())).to(self.dtype)
        if self.local:
            # Core ML does not clamp a negative slice start when the current
            # cache is shorter than the window. Form the nonnegative start.
            start = torch.clamp(torch._shape_as_tensor(keys)[2] - extent, min=0)
            keys, values = keys[:, :, start:], values[:, :, start:]
        return self.linear(output, prefix + "o_proj"), keys, values, history_k, history_v

    def mixture(self, value):
        c = self.config
        scores = torch.sigmoid(F.linear(value.float(), self.weight("mlp.gate_weight")))
        indices = torch.topk(scores + self.weight("mlp.e_score_correction_bias"), c["num_experts_per_tok"], dim=-1).indices
        probabilities = torch.gather(scores, -1, indices)
        probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20) * c["route_scale"]
        expanded = value.unsqueeze(-2)
        gate, up = self.gate(expanded, indices).float(), self.up(expanded, indices).float()
        hidden = (torch.tanh(gate / c["mlp_gate_softcap"]) * c["mlp_gate_softcap"] * torch.sigmoid(gate)
                  * torch.tanh(up / c["mlp_up_softcap"]) * c["mlp_up_softcap"]).to(self.dtype)
        output = self.down(hidden, indices)
        return (output.float() * probabilities.unsqueeze(-1)).sum(-2).to(self.dtype)

    def forward(self, x, blocks, bank, enabled, keys, values, valid, history_k, history_v, history_a, history_m, position):
        c, layer = self.config, self.layer
        messages = None
        if self.core:
            normalized = rms_norm(bank)
            scores = torch.matmul(self.weight("message_query"), normalized.transpose(-1, -2)) / math.sqrt(c["hidden_size"])
            messages = torch.matmul(torch.softmax(scores, dim=-1), normalized)
        mixed = x.unsqueeze(-2).float() if layer == 0 else residual_mix(
            blocks, x, self.weight("attention_input.query"), c["attn_res_normalize_values"])
        parts = []
        for slot, name in enumerate(["attn", "k", "v"]):
            query_count = 1 if self.local or layer == 0 else 3
            part = mixed[:, :, min(slot, query_count - 1)]
            if messages is not None:
                part = self.controller(part, messages[:, :, slot], "controllers." + name, enabled)
            norm = "input_layernorm" if self.local else ["q", "k", "v"][slot] + "_input_layernorm"
            parts.append(self.norm(part, norm))
        attention, keys, values, history_k, history_v = self.attention(
            *parts, keys, values, valid, history_k, history_v, position)
        partial, history_a = self.convolution(attention, history_a, "attn_sconv")
        partial = partial.float()
        if layer % c["n_layers_per_attn_res_block"] == 0:
            blocks = x.float().unsqueeze(-2) if layer == 0 else torch.cat((blocks, x.float().unsqueeze(-2)), dim=-2)
        else:
            partial = partial + x.float()
        hidden = residual_mix(blocks, partial, self.weight("mlp_input.query"), c["attn_res_normalize_values"])[:, :, 0]
        if messages is not None:
            hidden = self.controller(hidden, messages[:, :, 3], "controllers.mlp", enabled)
        mixture = self.mixture(self.norm(hidden, "post_attention_layernorm"))
        hidden, history_m = self.convolution(mixture, history_m, "mlp_sconv")
        return ((partial + hidden.float()).to(self.dtype), blocks, keys, values,
                history_k, history_v, history_a, history_m)


class OutputHead(nn.Module):
    def __init__(self, config, weights, *, quantized=True, dtype=torch.float16, group_size=64):
        super().__init__()
        self.config, self.dtype = config, dtype
        self.params = Parameters({name: weights["language_model.model." + name]
                                 for name in ["final_input.query", "norm.weight"]})
        matrix = torch.as_tensor(weights["language_model.lm_head.weight"])
        self.projection = (CompressedLinear(matrix, group_size, torch.float32) if quantized
                           else PlainLinear(matrix, torch.float32))

    def forward(self, x, blocks):
        c = self.config
        mixed = residual_mix(blocks, x, self.params.get("final_input.query"), c["attn_res_normalize_values"])[:, :, 0]
        hidden = rms_norm(mixed, self.params.get("norm.weight"), c["rms_norm_eps"]).to(self.dtype)
        logits = self.projection(hidden).float()
        if c["final_logit_softcapping"] > 0:
            logits = torch.tanh(logits / c["final_logit_softcapping"]) * c["final_logit_softcapping"]
        return logits[:, :, :c["unpadded_vocab_size"]]


class VisionLayer(nn.Module):
    def __init__(self, config, weights, dtype):
        super().__init__()
        self.config, self.dtype = config, dtype
        weights = dict(weights)
        self.linears = nn.ModuleDict()
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                     "mlp.fc1", "mlp.fc2"]:
            self.linears[name.replace(".", "_")] = PlainLinear(weights.pop(name + ".weight"), dtype,
                                                               weights.pop(name + ".bias"))
        self.params = Parameters(weights)

    def linear(self, value, name):
        return self.linears[name.replace(".", "_")](value)

    def norm(self, value, name):
        return F.layer_norm(value.float(), [self.config["hidden_size"]], self.params.get(name + ".weight"),
                            self.params.get(name + ".bias"), self.config["layer_norm_eps"]).to(self.dtype)

    def forward(self, value, cosine, sine):
        width, heads = self.config["hidden_size"], self.config["num_attention_heads"]
        dim = width // heads
        tokens = value.shape[0]
        normalized = self.norm(value, "layernorm_before")
        query = self.linear(normalized, "self_attn.q_proj").reshape(1, tokens, heads, dim)
        key = self.linear(normalized, "self_attn.k_proj").reshape(1, tokens, heads, dim)
        def rotary(tensor):
            pairs = tensor.reshape(1, tokens, heads, dim // 2, 2)
            rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).reshape_as(tensor)
            return (tensor.float() * cosine.reshape(1, -1, 1, dim)
                    + rotated.float() * sine.reshape(1, -1, 1, dim)).to(self.dtype).permute(0, 2, 1, 3)
        query, key = rotary(query), rotary(key)
        values = self.linear(normalized, "self_attn.v_proj").reshape(1, tokens, heads, dim).permute(0, 2, 1, 3)
        probabilities = torch.softmax(torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(dim), dim=-1)
        attention = torch.matmul(probabilities, values.float()).to(self.dtype).permute(0, 2, 1, 3).reshape(tokens, width)
        value = value + self.params.get("lambda_1").to(self.dtype) * self.linear(attention, "self_attn.out_proj")
        hidden = F.gelu(self.linear(self.norm(value, "layernorm_after"), "mlp.fc1"))
        return value + self.params.get("lambda_2").to(self.dtype) * self.linear(hidden, "mlp.fc2")


class Vision(nn.Module):
    def __init__(self, config, weights, dtype=torch.float16):
        super().__init__()
        self.config, self.dtype = config, dtype
        weights = {name.removeprefix("vision_tower."): value for name, value in weights.items()}
        self.patch = PlainLinear(weights.pop("embeddings.patch_embedding.weight"), dtype)
        self.project_in = PlainLinear(weights.pop("proj_in.weight"), dtype)
        self.project_out = PlainLinear(weights.pop("proj_out.weight"), dtype)
        self.layers = nn.ModuleList()
        for layer in range(config["num_hidden_layers"]):
            prefix = f"layers.{layer}."
            parameters = {name.removeprefix(prefix): value for name, value in weights.items() if name.startswith(prefix)}
            self.layers.append(VisionLayer(config, parameters, dtype))
        self.params = Parameters({name: value for name, value in weights.items() if not name.startswith("layers.")})

    def forward(self, pixels, position_indices, position_coefficients, cosine, sine):
        patch = self.config["native_patch_size"]
        merge = self.config["adapter_merge_size"]
        width = self.config["hidden_size"]
        # Fold the image and vertical merge axes to keep every Core ML tensor
        # within its five-dimensional rank limit.
        values = pixels.reshape(-1, patch, merge, patch, 3).permute(0, 2, 1, 3, 4)
        values = self.patch(values.reshape(-1, patch * patch * 3))
        positions = F.embedding(position_indices, self.params.get("embeddings.position_embedding"))
        positions = (positions * position_coefficients.unsqueeze(-1)).sum(-2).to(self.dtype)
        values = values + positions
        cls = (self.params.get("embeddings.class_embedding").to(self.dtype)
               + self.params.get("embeddings.cls_position_embedding").to(self.dtype)).reshape(1, width)
        values = torch.cat((cls, values), dim=0)
        values = F.layer_norm(values.float(), [width], self.params.get("pre_layernorm.weight"),
                              self.params.get("pre_layernorm.bias"), self.config["layer_norm_eps"]).to(self.dtype)
        for layer in self.layers:
            values = layer(values, cosine, sine)
        values = values[1:].reshape(pixels.shape[0], merge * merge * width)
        return self.project_out(F.gelu(self.project_in(values))).float()


def image_inputs(rgb, config, max_patches=1024):
    """Prepare normalized patches, bilinear positions, and two-axis rotary angles."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image")
    height, width = rgb.shape[:2]
    if not 0 < height <= 4096 or not 0 < width <= 4096:
        raise ValueError("Image dimensions must be in 1...4096")
    patch, merge = config["native_patch_size"], config["adapter_merge_size"]
    size = patch * merge
    rows, columns = (height + size - 1) // size, (width + size - 1) // size
    if rows * columns > max_patches:
        raise ValueError("Image exceeds the exported patch budget")
    pixels = np.full((rows * size, columns * size, 3), -1 / 127.5 - 1, dtype=np.float32)
    pixels[:height, :width] = rgb.astype(np.float32) / 127.5 - 1
    pixels = pixels.reshape(rows, size, columns, size, 3).transpose(0, 2, 1, 3, 4)
    grid = config["posemb_grid_size"]
    indices, coefficients, coordinates = [], [], [[0, 0]]
    for row in range(rows):
        for column in range(columns):
            for dy in range(merge):
                for dx in range(merge):
                    y, x = row * merge + dy, column * merge + dx
                    yf = max(0, (y + .5) * grid / (rows * merge) - .5)
                    xf = max(0, (x + .5) * grid / (columns * merge) - .5)
                    y0, x0 = min(int(yf), grid - 1), min(int(xf), grid - 1)
                    y1, x1 = min(y0 + 1, grid - 1), min(x0 + 1, grid - 1)
                    wy, wx = yf - y0, xf - x0
                    indices.append([y0 * grid + x0, y0 * grid + x1, y1 * grid + x0, y1 * grid + x1])
                    coefficients.append([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
                    coordinates.append([x + 1, y + 1])
    dim = config["hidden_size"] // config["num_attention_heads"]
    inverse = np.power(10000., -np.arange(0, dim // 2, 2, dtype=np.float32) / (dim // 2))
    angles = (np.array(coordinates, dtype=np.float32)[..., None] * inverse).reshape(-1, dim // 2)
    angles = np.repeat(angles, 2, axis=-1)
    return {"pixels": pixels.reshape(-1, size, size, 3),
            "position_indices": np.array(indices, dtype=np.int32),
            "position_coefficients": np.array(coefficients, dtype=np.float32),
            "cosine": np.cos(angles), "sine": np.sin(angles)}


def layer_visits(config):
    loop = config["loop_config"]
    return (list(range(loop["core_start"]))
            + list(range(loop["core_start"], loop["core_end"])) * loop["num_passes"]
            + list(range(loop["core_end"], config["num_hidden_layers"])))


class Decoder:
    """Partition orchestration shared by the exporter's eager and Core ML checks."""

    def __init__(self, config, embedding, layers, head, dtype=torch.float16):
        self.config, self.embedding, self.layers, self.head = config, embedding, layers, head
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.position = 0
        self.states = [None for _ in layer_visits(self.config)]

    def forward(self, input_ids, embeddings=None, produce_logits=True):
        c = self.config
        x = self.embedding(input_ids) if embeddings is None else embeddings
        count, width = input_ids.shape[1], c["hidden_size"]
        blocks = torch.zeros(1, count, 1, width)
        bank = None
        prelude = prelude_blocks = None
        for visit, layer in enumerate(layer_visits(c)):
            if layer == c["loop_config"]["core_start"]:
                if visit == layer:
                    prelude, prelude_blocks = x, blocks
                else:
                    bank = torch.cat((blocks[:, :, prelude_blocks.shape[-2] + 1:], x.float().unsqueeze(-2)), dim=-2)
                    x, blocks = prelude, prelude_blocks
            local = layer in c["local_layer_ids"]
            kv = c["swa_num_key_value_heads"] if local else c["num_key_value_heads"]
            dim = c["swa_head_dim"] if local else c["head_dim"]
            initial = self.states[visit] is None
            if initial:
                state = [torch.zeros(1, kv, 1, dim, dtype=self.dtype) for _ in range(2)]
                state += [torch.zeros(1, c["sconv_kernel_size"] - 1, channels)
                          for channels in [kv * dim, kv * dim, width, width]]
            else:
                state = self.states[visit]
            valid = torch.zeros(state[0].shape[2]) if initial else torch.ones(state[0].shape[2])
            enabled = torch.tensor([float(bank is not None and layer < c["loop_config"]["core_end"])])
            messages = bank if bank is not None else torch.zeros(1, count, 1, width)
            result = self.layers[layer](x, blocks, messages, enabled, state[0], state[1], valid,
                                        *state[2:], torch.tensor([self.position], dtype=torch.int32))
            x, blocks, keys, values, *histories = result[:8]
            if initial and (not local or count < c["sliding_window_size"]):
                keys, values = keys[:, :, 1:], values[:, :, 1:]
            self.states[visit] = [keys, values, *histories]
        self.position += count
        return self.head(x, blocks) if produce_logits else None


LAYER_INPUTS = ["x", "blocks", "bank", "enabled", "keys", "values", "valid",
                "history_k", "history_v", "history_a", "history_m", "position"]
LAYER_OUTPUTS = ["x_out", "blocks_out", "keys_out", "values_out", "history_k_out",
                 "history_v_out", "history_a_out", "history_m_out"]


def conversion_inputs(config, kind, *, layer=0, context_limit=8192, prefill_size=128,
                      dtype=torch.float16, max_patches=1024):
    import coremltools as ct
    precision = np.float16 if dtype == torch.float16 else np.float32
    if kind == "vision":
        merge = config["adapter_merge_size"]
        size = merge * config["native_patch_size"]
        dim = config["hidden_size"] // config["num_attention_heads"]
        default = min(2, max_patches)
        patches = ct.RangeDim(1, max_patches, default=default)
        subpatches = ct.RangeDim(merge * merge, max_patches * merge * merge,
                                default=default * merge * merge)
        with_class = ct.RangeDim(merge * merge + 1, max_patches * merge * merge + 1,
                                default=default * merge * merge + 1)
        specs = [ct.TensorType(name="pixels", shape=(patches, size, size, 3), dtype=np.float32),
                 ct.TensorType(name="position_indices", shape=(subpatches, 4), dtype=np.int32),
                 ct.TensorType(name="position_coefficients", shape=(subpatches, 4), dtype=np.float32),
                 ct.TensorType(name="cosine", shape=(with_class, dim), dtype=np.float32),
                 ct.TensorType(name="sine", shape=(with_class, dim), dtype=np.float32)]
        sample = image_inputs(np.zeros((size, size * default, 3), dtype=np.uint8), config, max_patches)
        return specs, sample, ["image_embeddings"]
    tokens = 1 if prefill_size == 1 else ct.RangeDim(1, prefill_size, default=min(2, prefill_size))
    width = config["hidden_size"]
    rng = np.random.default_rng(97)
    def floats(shape, floating=precision):
        return (rng.normal(0, .1, shape)).astype(floating)
    if kind == "embedding":
        return [ct.TensorType(name="input_ids", shape=(1, tokens), dtype=np.int32)], \
            {"input_ids": np.ones((1, min(2, prefill_size)), dtype=np.int32)}, ["embeddings"]
    max_blocks = config["num_hidden_layers"] // config["n_layers_per_attn_res_block"]
    blocks = ct.RangeDim(1, max_blocks, default=min(2, max_blocks))
    count = min(2, prefill_size)
    specs = [ct.TensorType(name="x", shape=(1, tokens, width), dtype=precision),
             ct.TensorType(name="blocks", shape=(1, tokens, blocks, width), dtype=np.float32)]
    sample = {"x": floats((1, count, width)),
              "blocks": floats((1, count, min(2, max_blocks), width), np.float32)}
    if kind == "head":
        return specs, sample, ["logits"]
    if kind != "layer":
        raise ValueError("Unknown conversion part")
    local = layer in config["local_layer_ids"]
    kv = config["swa_num_key_value_heads"] if local else config["num_key_value_heads"]
    dim = config["swa_head_dim"] if local else config["head_dim"]
    max_history = min(context_limit, config["sliding_window_size"]) if local else context_limit
    history = ct.RangeDim(1, max_history, default=min(3, max_history))
    bank = ct.RangeDim(1, max_blocks, default=min(2, max_blocks))
    specs += [ct.TensorType(name="bank", shape=(1, tokens, bank, width), dtype=np.float32),
              ct.TensorType(name="enabled", shape=(1,), dtype=np.float32),
              ct.TensorType(name="keys", shape=(1, kv, history, dim), dtype=precision),
              ct.TensorType(name="values", shape=(1, kv, history, dim), dtype=precision),
              ct.TensorType(name="valid", shape=(history,), dtype=np.float32)]
    sample.update({"bank": floats((1, count, min(2, max_blocks), width), np.float32),
                   "enabled": np.ones((1,), dtype=np.float32),
                   "keys": floats((1, kv, min(3, max_history), dim)),
                   "values": floats((1, kv, min(3, max_history), dim)),
                   "valid": np.ones((min(3, max_history),), dtype=np.float32)})
    for name, channels in zip(["history_k", "history_v", "history_a", "history_m"],
                              [kv * dim, kv * dim, width, width]):
        shape = (1, config["sconv_kernel_size"] - 1, channels)
        specs.append(ct.TensorType(name=name, shape=shape, dtype=np.float32))
        sample[name] = floats(shape, np.float32)
    specs.append(ct.TensorType(name="position", shape=(1,), dtype=np.int32))
    sample["position"] = np.array([3], dtype=np.int32)
    return specs, sample, LAYER_OUTPUTS


def convert_part(module, config, kind, output, *, layer=0, context_limit=8192,
                 prefill_size=128, dtype=torch.float16, max_patches=1024):
    import coremltools as ct
    from coremltools.converters.mil.mil import types
    from collections import Counter
    module.eval()
    specs, sample, names = conversion_inputs(config, kind, layer=layer, context_limit=context_limit,
                                             prefill_size=prefill_size, dtype=dtype, max_patches=max_patches)
    with torch.no_grad():
        traced = torch.jit.trace(module, tuple(torch.from_numpy(value) for value in sample.values()), check_trace=False)
    outputs = []
    for name in names:
        output_dtype = np.float32
        if name in ["embeddings", "x_out", "keys_out", "values_out"]:
            output_dtype = np.float16 if dtype == torch.float16 else np.float32
        outputs.append(ct.TensorType(name=name, dtype=output_dtype))
    converted = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
        inputs=specs, outputs=outputs, compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    operations = list(converted._mil_program.functions["main"].operations)
    integer_gathers = [op.name for op in operations if op.op_type == "gather"
                      and types.builtin_to_string(op.x.dtype) == "int8"]
    expected = 3 if isinstance(module, DecoderLayer) and isinstance(module.gate, Int8Experts) else 0
    if len(integer_gathers) < expected:
        raise ValueError("Conversion expanded expert or embedding weights before their INT8 gather")
    if any(isinstance(part, CompressedLinear) for part in module.modules()):
        if not any(op.op_type.startswith("constexpr_") for op in operations):
            raise ValueError("Core ML dense INT8 compression metadata was lost")
    converted.save(str(output))
    # coremltools otherwise keeps its temporary package until process exit.
    # Reclaim it after saving each partition instead of accumulating a second
    # complete checkpoint during the streamed export.
    temporary_package = Path(converted.package_path)
    if converted.is_temp_package and temporary_package.resolve() != Path(output).resolve():
        shutil.rmtree(temporary_package)
    return {"inputs": list(sample), "outputs": names, "int8_gathers": integer_gathers,
            "operations": dict(Counter(op.op_type for op in operations))}
