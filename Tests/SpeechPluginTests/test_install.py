import contextlib
import fcntl
import importlib.util
import io
import json
import stat
import tempfile
import unittest
import zipfile
from unittest import mock
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "plugins/soniqo-speech/scripts/install-local.py"
SPEC = importlib.util.spec_from_file_location("install_speech", SCRIPT)
INSTALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL)


class SpeechPluginInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name) / "home"
        self.home.mkdir()
        self.archive = Path(self.temporary.name) / "plugin.zip"
        with zipfile.ZipFile(self.archive, "w") as package:
            package.writestr("plugin.json", json.dumps({"name": "soniqo-speech", "version": "0.1.0"}))
            package.writestr("dist/server.cjs", "// test fixture")
            native = zipfile.ZipInfo("native/speech")
            native.external_attr = (stat.S_IFREG | 0o755) << 16
            package.writestr(native, "#!/bin/sh\nexit 0\n")

    def install(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return INSTALL.install(self.archive, self.home)

    def test_installs_built_package_and_preserves_native_execution_permission(self):
        destination = self.install()
        self.assertTrue((destination / "dist/server.cjs").is_file())
        self.assertEqual((destination / "native/speech").stat().st_mode & 0o777, 0o755)
        catalog = json.loads((self.home / ".agents/plugins/marketplace.json").read_text())
        self.assertEqual(catalog["plugins"][0]["source"]["path"], "./.codex/plugins/local/soniqo-speech")

    def test_preserves_existing_marketplace_identity_entries_and_backup(self):
        catalog_path = self.home / ".agents/plugins/marketplace.json"
        catalog_path.parent.mkdir(parents=True)
        original = {"name": "personal", "plugins": [{"name": "other", "source": {"path": "./other"}}]}
        catalog_path.write_text(json.dumps(original))
        self.install()
        catalog = json.loads(catalog_path.read_text())
        self.assertEqual(catalog["name"], "personal")
        self.assertEqual(catalog["plugins"][0], original["plugins"][0])
        self.assertEqual(json.loads(catalog_path.with_suffix(".json.speech-backup").read_text()), original)

    def test_rejects_unsafe_archive_paths_before_creating_a_plugin(self):
        with zipfile.ZipFile(self.archive, "a") as package:
            package.writestr("../outside", "unsafe")
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            self.install()
        self.assertFalse((self.home / ".codex/plugins/local/soniqo-speech").exists())
        self.assertFalse((self.home / "outside").exists())

    def test_existing_backups_are_preserved_when_installing_an_update(self):
        destination = self.install()
        marker = destination / "keep"
        marker.write_text("existing source")
        catalog_path = self.home / ".agents/plugins/marketplace.json"
        catalog_path.with_suffix(".json.speech-backup").write_text("prior backup")
        self.install()
        self.assertEqual(catalog_path.with_suffix(".json.speech-backup").read_text(), "prior backup")
        self.assertTrue(catalog_path.with_name("marketplace.json.speech-backup.1").exists())
        self.assertEqual((destination.with_name("soniqo-speech.previous") / "keep").read_text(), "existing source")

    def test_rejects_normalized_and_case_colliding_paths(self):
        original = self.archive.read_bytes()
        for name in ("./plugin.json", "dist//server.cjs", "dist/Server.cjs"):
            with self.subTest(name=name):
                self.archive.write_bytes(original)
                with zipfile.ZipFile(self.archive, "a") as package:
                    package.writestr(name, "synthetic replacement")
                with self.assertRaisesRegex(ValueError, "unsafe path|duplicate paths"):
                    self.install()
                self.assertFalse((self.home / ".codex/plugins/local/soniqo-speech").exists())

    def test_rejects_archive_symlinks_and_preserves_the_existing_source(self):
        destination = self.install()
        marker = destination / "keep"
        marker.write_text("unchanged")
        with zipfile.ZipFile(self.archive, "a") as package:
            link = zipfile.ZipInfo("native/link")
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(link, "/synthetic-private-data")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.install()
        self.assertEqual(marker.read_text(), "unchanged")

    def test_refuses_concurrent_installations(self):
        directory = self.home / ".agents/plugins"
        directory.mkdir(parents=True)
        with (directory / ".speech-install.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "Another Speech installation"):
                self.install()

    def test_concurrent_catalog_edits_are_preserved_and_the_source_is_rolled_back(self):
        destination = self.install()
        marker = destination / "keep"
        marker.write_text("unchanged")
        catalog_path = self.home / ".agents/plugins/marketplace.json"
        changed = json.loads(catalog_path.read_text())
        changed["plugins"].append({"name": "other"})
        replace = Path.rename

        def change_after_source_swap(source, target):
            result = replace(source, target)
            if source.name == "plugin":
                catalog_path.write_text(json.dumps(changed))
            return result

        with mock.patch.object(Path, "rename", change_after_source_swap):
            with self.assertRaisesRegex(ValueError, "marketplace changed"):
                self.install()
        self.assertEqual(json.loads(catalog_path.read_text()), changed)
        self.assertEqual(marker.read_text(), "unchanged")

    def test_does_not_replace_a_symlinked_marketplace(self):
        catalog_path = self.home / ".agents/plugins/marketplace.json"
        catalog_path.parent.mkdir(parents=True)
        target = self.home / "other-catalog.json"
        target.write_text(json.dumps({"name": "other", "plugins": []}))
        catalog_path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinked marketplace"):
            self.install()
        self.assertTrue(catalog_path.is_symlink())
        self.assertEqual(json.loads(target.read_text())["plugins"], [])


if __name__ == "__main__":
    unittest.main()
