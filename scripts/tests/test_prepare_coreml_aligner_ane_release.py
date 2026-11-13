"""Neural Engine release layout, cache isolation and publication provenance."""
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import coremltools as ct

from test_prepare_coreml_aligner_cpu_release import ReleaseFixture, release
from test_prepare_coreml_aligner_cpu_probe import operations, probe, mil


class ANEReleaseTests(ReleaseFixture):
    def test_ane_release_preserves_gpu_and_cpu_bundles_and_source(self):
        cpu = self.base / "text_decoder_cpu_v1.mlmodelc/model.mil"
        cpu.parent.mkdir()
        cpu.write_bytes(b"published CPU graph")
        manifest = self.prepare(target="ane")
        root = self.root / "release/int8"
        self.assertEqual({p.name for p in (root / "upload").iterdir()}, {
            "text_decoder_ane_v1.mlmodelc", "text_decoder_ane_v1.mlpackage",
            "README.md", "ane_decoder_v1.json"})
        for location in [self.base, root / "validation"]:
            self.assertEqual((location / "text_decoder.mlmodelc/model.mil").read_bytes(),
                             b"original GPU graph")
            self.assertEqual((location / "text_decoder_cpu_v1.mlmodelc/model.mil").read_bytes(),
                             b"published CPU graph")
        self.assertEqual((self.source / "Data/com.apple.CoreML/model.mlmodel").read_bytes(),
                         self.source_bytes)
        self.assertFalse(manifest["legacy_gpu_decoder_modified"])
        self.assertFalse(manifest["cpu_compatible_decoder_modified"])

    def test_ane_specification_squares_with_mul_and_records_source(self):
        manifest = self.prepare(target="ane")
        root = self.root / "release/int8/upload"
        spec = ct.utils.load_spec(str(root / "text_decoder_ane_v1.mlpackage/Data/com.apple.CoreML/model.mlmodel"))
        square = operations(spec)["square"]
        self.assertEqual(square.type, "mul")
        self.assertEqual([probe.reference(square, key) for key in ("x", "y")], ["x", "x"])
        self.assertEqual(square.outputs[0].type.tensorType.dataType, mil.FLOAT16)
        metadata = spec.description.metadata.userDefined
        self.assertEqual(metadata["ane_normalization_square"], "mul")
        self.assertEqual(metadata["ane_decoder_bundle_version"], "1")
        self.assertEqual(metadata["source_model_revision"], self.metadata["revision"])
        self.assertNotIn("cpu_normalization_precision", metadata)
        self.assertEqual(manifest["normalization_precision"], "float16")
        self.assertEqual(manifest["normalization_square"], "mul")
        self.assertEqual(manifest["publication_base_revision"], "b" * 40)

    def test_ane_weights_match_source_compiled_model_and_manifest(self):
        manifest = self.prepare(target="ane")
        root = self.root / "release/int8/upload"
        weights = [item for item in manifest["files"] if item["path"].endswith("weight.bin")]
        self.assertEqual(len(weights), 2)
        for item in weights:
            self.assertEqual(item["sha256"], self.weight_hash)
            self.assertEqual((root / item["path"]).read_bytes(), b"unchanged learned weights")

    def test_ane_card_preserves_existing_cpu_instructions(self):
        self.prepare(target="ane")
        card = (self.root / "release/int8/upload/README.md").read_text()
        self.assertIn("Existing CPU instructions.", card)
        self.assertIn("text_decoder_ane_v1", card)
        self.assertIn("legacy offline cache", card)

    def test_changed_compiler_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed learned weights"):
            self.prepare(target="ane", corrupt_compile=True)

    def test_existing_ane_release_directory_is_not_overwritten(self):
        self.prepare(target="ane")
        with self.assertRaisesRegex(ValueError, "output already exists"):
            self.prepare(target="ane")

    def test_current_card_pins_the_published_revision(self):
        info = {"sha": "c" * 40, "private": False,
                "siblings": [{"rfilename": "text_decoder_cpu_v1.mlmodelc/model.mil"}]}
        with patch.object(release, "urlopen", return_value=io.BytesIO(json.dumps(info).encode())), \
             patch.object(release, "pinned_model_card", return_value="CPU documentation") as fetch:
            card, revision = release.current_model_card(self.metadata)
        self.assertEqual((card, revision), ("CPU documentation", "c" * 40))
        self.assertEqual(fetch.call_args.args[0]["revision"], "c" * 40)

    def test_published_ane_bundle_cannot_be_replaced_in_place(self):
        info = {"sha": "c" * 40, "private": False,
                "siblings": [{"rfilename": "text_decoder_ane_v1.mlmodelc/model.mil"}]}
        with patch.object(release, "urlopen", return_value=io.BytesIO(json.dumps(info).encode())), \
             patch.object(release, "pinned_model_card") as fetch:
            with self.assertRaisesRegex(ValueError, "already exists"):
                release.current_model_card(self.metadata)
            fetch.assert_not_called()

    def test_cli_target_reaches_preparation_and_reports_the_selected_bundle(self):
        result = {"repo": "example/model", "decoder_bundle": "text_decoder_ane_v1", "files": []}
        output = io.StringIO()
        with patch.object(release, "prepare_variant", return_value=result) as prepare, \
             patch.dict(release.VARIANTS, {"int8": self.metadata}, clear=True), redirect_stdout(output):
            release.main(["--target", "ane", "--source-root", str(self.root / "source"),
                          "--cache-root", str(self.root / "cache"), "--output-root", str(self.root / "cli")])
        self.assertEqual(prepare.call_args.kwargs["target"], "ane")
        self.assertEqual(json.loads(output.getvalue())["decoder_bundle"], "text_decoder_ane_v1")
        self.assertEqual(json.loads((self.root / "cli/release-manifest.json").read_text()), [result])


if __name__ == "__main__":
    unittest.main()
