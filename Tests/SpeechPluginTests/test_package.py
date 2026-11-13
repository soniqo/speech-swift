import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "plugins/soniqo-speech"
SPEC = importlib.util.spec_from_file_location("speech_package", ROOT / "scripts/package.py")
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class SpeechPluginPackageTests(unittest.TestCase):
    def setUp(self):
        names = ["plugin.json", "mcp.json", "skills/speech/SKILL.md",
                 "assets/logo.png", "assets/logo-dark.png", "dist/cli-catalog.json"]
        self.files = {name: (ROOT / name).read_bytes() for name in names}

    def change_manifest(self, update):
        manifest = json.loads(self.files["plugin.json"])
        update(manifest)
        self.files["plugin.json"] = json.dumps(manifest).encode()

    def test_development_package_reports_real_submission_gaps(self):
        manifest, gaps = PACKAGE.validate(self.files)
        self.assertEqual(manifest["name"], "soniqo-speech")
        self.assertTrue(any("privacyPolicyURL" in item for item in gaps))
        self.assertTrue(any("recording" in item for item in gaps))

    def test_public_mode_rejects_missing_materials(self):
        with self.assertRaisesRegex(ValueError, "Not ready"):
            PACKAGE.validate(self.files, submission=True)

    def test_rejects_missing_referenced_icon(self):
        del self.files["assets/logo-dark.png"]
        with self.assertRaisesRegex(ValueError, "Missing icon"):
            PACKAGE.validate(self.files)

    def test_rejects_non_png_and_small_icons(self):
        self.files["assets/logo.png"] = b"<svg>not a PNG</svg>"
        with self.assertRaisesRegex(ValueError, "PNG"):
            PACKAGE.validate(self.files)

    def test_rejects_private_app_references(self):
        self.change_manifest(lambda manifest: manifest["extensions"]["com.openai"].update(
            {"apps": "./.app.json"}))
        with self.assertRaisesRegex(ValueError, "private app"):
            PACKAGE.validate(self.files)

    def test_rejects_an_unexpected_server_command(self):
        original = json.loads(self.files["mcp.json"])
        for command in ("/bin/bash", "/bin/sh", "bash"):
            with self.subTest(command=command):
                mcp = json.loads(json.dumps(original))
                mcp["mcpServers"]["speech"]["command"] = command
                self.files["mcp.json"] = json.dumps(mcp).encode()
                with self.assertRaisesRegex(ValueError, "transport"):
                    PACKAGE.validate(self.files)

    def test_listing_matches_confirmed_availability_and_commerce(self):
        extension = json.loads(self.files["plugin.json"])["extensions"]["com.openai"]
        self.assertEqual(extension["publication"]["countries"], [])
        self.assertIs(extension["review"]["commerce"], False)
        self.assertEqual(len(extension["review"]["test_cases"]["positive"]), 5)
        self.assertEqual(len(extension["review"]["test_cases"]["negative"]), 3)

    def test_rejects_an_unmatched_packaged_cli_binary(self):
        self.files["native/speech"] = b"a different runtime"
        with self.assertRaisesRegex(ValueError, "does not match"):
            PACKAGE.validate(self.files)

    def test_rejects_a_missing_cli_catalogue(self):
        del self.files["dist/cli-catalog.json"]
        with self.assertRaisesRegex(ValueError, "catalogue is missing"):
            PACKAGE.validate(self.files)

    def test_packaging_rejects_symlinked_files_without_copying_their_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "synthetic-private-data"
            private.write_text("SYNTHETIC TEST CONTENT ONLY")
            link = root / "mlx.metallib"
            link.symlink_to(private)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                PACKAGE.copy_regular(link, root / "copied")
            self.assertFalse((root / "copied").exists())

    def test_packaging_rejects_symlinked_component_roots_and_nested_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "component"
            source.mkdir()
            private = root / "synthetic-private-data"
            private.write_text("SYNTHETIC TEST CONTENT ONLY")
            (source / "nested").symlink_to(private)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                PACKAGE.copy_tree(source, root / "stage")
            self.assertFalse((root / "stage").exists())
            (source / "nested").unlink()
            link = root / "linked-component"
            link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                PACKAGE.copy_tree(link, root / "stage")

    def test_native_packaging_rejects_unreviewed_adjacent_bundles_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "speech").write_text("not executable")
            (root / "mlx.metallib").write_text("test shader")
            for name in PACKAGE.RUNTIME_BUNDLES | {"Unreviewed.bundle"}:
                (root / name).mkdir()
            with self.assertRaisesRegex(ValueError, "reviewed allowlist"):
                PACKAGE.add_native_runtime(root / "stage", root, "0.0.29")


if __name__ == "__main__":
    unittest.main()
