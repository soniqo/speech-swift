"""Release layout and weight-preservation checks without inference/downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import coremltools as ct

from test_prepare_coreml_aligner_cpu_probe import graph_fixture

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import prepare_coreml_aligner_cpu_release as release
from prepare_coreml_aligner_cpu_probe import transform


class CPUReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source/int8/text_decoder.mlpackage"
        model = self.source / "Data/com.apple.CoreML/model.mlmodel"
        model.parent.mkdir(parents=True)
        ct.utils.save_spec(graph_fixture(), str(model))
        (self.source / "Manifest.json").write_text("{}")
        source_weights = self.source / "Data/com.apple.CoreML/weights/weight.bin"
        source_weights.parent.mkdir()
        source_weights.write_bytes(b"unchanged learned weights")
        self.base = self.root / "cache/int8"
        weights = self.base / "text_decoder.mlmodelc/weights/weight.bin"
        weights.parent.mkdir(parents=True)
        shutil.copyfile(source_weights, weights)
        (self.base / "text_decoder.mlmodelc/model.mil").write_bytes(b"original GPU graph")
        (self.base / "config.json").write_text("{}")
        self.weight_hash = hashlib.sha256(source_weights.read_bytes()).hexdigest()
        self.source_bytes = model.read_bytes()
        self.metadata = {"repo": "example/model", "revision": "a" * 40,
                         "weight_sha256": self.weight_hash}

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, corrupt_compile=False):
        compiled = self.root / "compiler-output.mlmodelc"

        def compile_model(package):
            compiled.mkdir()
            (compiled / "weights").mkdir()
            data = (Path(package) / "Data/com.apple.CoreML/weights/weight.bin").read_bytes()
            (compiled / "weights/weight.bin").write_bytes(b"modified" if corrupt_compile else data)
            (compiled / "model.mil").write_bytes(b"repaired CPU graph")
            return str(compiled)

        with patch.dict(release.VARIANTS, {"int8": self.metadata}), \
             patch.object(release, "pinned_model_card", return_value="Original model card\n"), \
             patch.object(release, "transform", side_effect=lambda spec, mode: transform(spec, mode, expected_norms=1)), \
             patch.object(release.ct.models.utils, "compile_model", side_effect=compile_model):
            return release.prepare_variant("int8", self.root / "source", self.root / "cache",
                                           self.root / "release")

    def test_release_is_additive_and_validation_inputs_cannot_be_uploaded(self):
        manifest = self.prepare()
        root = self.root / "release/int8"
        self.assertEqual(set(path.name for path in (root / "upload").iterdir()), {
            "text_decoder_cpu_v1.mlmodelc", "text_decoder_cpu_v1.mlpackage",
            "README.md", "cpu_decoder_v1.json"})
        self.assertNotIn("text_decoder.mlmodelc", [entry["path"] for entry in manifest["files"]])
        self.assertEqual((root / "validation/text_decoder.mlmodelc/model.mil").read_bytes(),
                         b"original GPU graph")
        self.assertEqual((self.base / "text_decoder.mlmodelc/model.mil").read_bytes(),
                         b"original GPU graph")
        self.assertEqual((self.source / "Data/com.apple.CoreML/model.mlmodel").read_bytes(),
                         self.source_bytes)
        self.assertFalse((self.root / "compiler-output.mlmodelc").exists())

    def test_source_and_compiled_cpu_weights_are_identical_and_manifested(self):
        manifest = self.prepare()
        root = self.root / "release/int8/upload"
        entries = [item for item in manifest["files"] if item["path"].endswith("weight.bin")]
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(item["sha256"] == self.weight_hash for item in entries))
        self.assertTrue(all((root / item["path"]).read_bytes() == b"unchanged learned weights"
                            for item in entries))
        published = json.loads((root / "cpu_decoder_v1.json").read_text())
        self.assertEqual(published["decoder_bundle"], "text_decoder_cpu_v1")
        self.assertFalse(published["legacy_gpu_decoder_modified"])

    def test_repaired_specification_records_its_precision_and_source(self):
        self.prepare()
        model = self.root / "release/int8/upload/text_decoder_cpu_v1.mlpackage/Data/com.apple.CoreML/model.mlmodel"
        metadata = ct.utils.load_spec(str(model)).description.metadata.userDefined
        self.assertEqual(metadata["cpu_normalization_precision"], "float32")
        self.assertEqual(metadata["source_model_revision"], self.metadata["revision"])
        self.assertEqual(metadata["cpu_decoder_bundle_version"], "1")

    def test_changed_compiler_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed learned weights"):
            self.prepare(corrupt_compile=True)

    def test_existing_release_directory_is_not_overwritten(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.prepare()

    def test_model_card_keeps_verified_public_links_and_removes_unverified_references(self):
        original = ("Public: https://github.com/soniqo/speech-swift\n"
                    "Unverified project: https://github.com/example/unverified-project\n")
        card = release.cpu_model_card(original)
        self.assertIn("https://github.com/soniqo/speech-swift", card)
        self.assertNotIn("Unverified project", card)
        self.assertNotIn("unverified-project", card)


if __name__ == "__main__":
    unittest.main()
