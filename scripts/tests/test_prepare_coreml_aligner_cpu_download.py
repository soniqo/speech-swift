"""Exercise the manual CPU release download with tiny in-memory responses."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CPUReleaseDownloadTests(unittest.TestCase):
    def run_download(self, corrupt=False, traversal=False, revision=None, oversized=False):
        workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/nightly-e2e.yml").read_text()
        section = workflow.split("- name: Download versioned CPU decoders at verified model commits", 1)[1]
        source = section.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
        source = "\n".join(line[10:] for line in source.splitlines())
        files = {name: name.encode() for name in [
            "config.json", "vocab.json", "merges.txt", "tokenizer_config.json", "embed_tokens.fp16.bin",
            "audio_encoder.mlmodelc/weights/weight.bin", "text_decoder_cpu_v1.mlmodelc/model.mil",
            "text_decoder_cpu_v1.mlmodelc/weights/weight.bin", "text_decoder.mlmodelc/model.mil"]}
        tree = []
        for name, data in files.items():
            entry = {"type": "file", "path": name, "size": len(data),
                     "oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
            if name.endswith("weight.bin"):
                entry["lfs"] = {"oid": hashlib.sha256(data).hexdigest()}
            tree.append(entry)
        if traversal:
            tree.insert(0, {"type": "file", "path": "text_decoder_cpu_v1.mlmodelc/../../escape",
                            "size": 1, "oid": "invalid"})
        if oversized:
            tree[0]["size"] = 3_200_000_001
        requested = []

        def fetch(url, timeout):
            requested.append(url)
            if "/api/models/" in url:
                return io.BytesIO(json.dumps(tree).encode())
            path = url.split("/resolve/", 1)[1].split("/", 1)[1]
            self.assertNotEqual(path, "text_decoder.mlmodelc/model.mil")
            data = files[path]
            return io.BytesIO(b"x" * len(data) if corrupt else data)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"RUNNER_TEMP": directory, "GITHUB_ENV": str(root / "environment"),
                   "INT8_REVISION": revision or "a" * 40, "FP16_REVISION": "b" * 40}
            with patch.dict(os.environ, env), patch("urllib.request.urlopen", fetch):
                exec(compile(source, "cpu-release-download", "exec"), {})
            for variant in ["int8", "fp16"]:
                downloaded = root / "aligner-cpu-release" / variant
                self.assertTrue((downloaded / "text_decoder_cpu_v1.mlmodelc/model.mil").exists())
                self.assertFalse((downloaded / "text_decoder.mlmodelc").exists())
            self.assertEqual(len(requested), 18)

    def test_both_pinned_variants_download_without_legacy_decoder(self):
        self.run_download()

    def test_same_size_corrupt_content_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.run_download(corrupt=True)

    def test_remote_path_traversal_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.run_download(traversal=True)

    def test_unpinned_revision_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "immutable model commit"):
            self.run_download(revision="main")

    def test_excessive_download_is_rejected_before_file_transfer(self):
        with self.assertRaisesRegex(AssertionError, "bounded download size"):
            self.run_download(oversized=True)


if __name__ == "__main__":
    unittest.main()
