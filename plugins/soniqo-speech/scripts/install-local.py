#!/usr/bin/env python3
"""Register a validated package in the user's personal desktop marketplace."""

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath


def unused_backup(path):
    candidate = path
    index = 0
    while candidate.exists() or candidate.is_symlink():
        index += 1
        candidate = path.with_name(path.name + "." + str(index))
    return candidate


@contextlib.contextmanager
def installation_lock(home):
    directory = home / ".agents/plugins"
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".speech-install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("The installer lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another Speech installation is running; retry after it finishes") from None
        yield
    finally:
        os.close(descriptor)


def catalog_bytes(path):
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlinked marketplace catalog")
    return path.read_bytes() if path.exists() else None


def validated_members(package):
    members = package.infolist()
    if len(members) > 10000 or sum(info.file_size for info in members) > 512 * 1024 * 1024:
        raise ValueError("Archive exceeds the package size limit")
    seen = set()
    for info in members:
        name = info.filename.removesuffix("/") if info.is_dir() else info.filename
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
            raise ValueError("Archive contains an unsafe path")
        identity = unicodedata.normalize("NFC", name).casefold()
        if identity in seen:
            raise ValueError("Archive contains duplicate paths")
        seen.add(identity)
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Archive must not contain symlinks or special files")
    return members


def install(archive, home):
    with installation_lock(home):
        return install_locked(archive, home)


def install_locked(archive, home):
    destination = home / ".codex/plugins/local/soniqo-speech"
    catalog_path = home / ".agents/plugins/marketplace.json"
    original_catalog = catalog_bytes(catalog_path)
    if original_catalog is not None:
        catalog = json.loads(original_catalog)
        if not isinstance(catalog, dict) or not isinstance(catalog.get("plugins"), list):
            raise ValueError("Existing marketplace is not a supported catalog; no files were changed")
    else:
        catalog = {"name": "soniqo-personal", "interface": {"displayName": "Soniqo Local"}, "plugins": []}
    entry = {
        "name": "soniqo-speech",
        "source": {"source": "local", "path": "./.codex/plugins/local/soniqo-speech"},
        "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
        "category": "Productivity",
    }
    matches = [index for index, plugin in enumerate(catalog["plugins"]) if plugin.get("name") == entry["name"]]
    if len(matches) > 1:
        raise ValueError("The marketplace contains duplicate Speech entries; no files were changed")
    if matches:
        catalog["plugins"][matches[0]] = {**catalog["plugins"][matches[0]], **entry}
    else:
        catalog["plugins"].append(entry)
    catalog_backup = unused_backup(catalog_path.with_suffix(".json.speech-backup"))
    backup = unused_backup(destination.with_name("soniqo-speech.previous"))
    with zipfile.ZipFile(archive) as package:
        members = validated_members(package)
        names = [info.filename for info in members]
        manifest = json.loads(package.read("plugin.json"))
        if manifest.get("name") != "soniqo-speech" or "dist/server.cjs" not in names:
            raise ValueError("This is not a built Soniqo Speech package")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="speech-install-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "plugin"
            stage.mkdir()
            for info in members:
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError("Archive must not contain symlinks")
                target = stage / info.filename
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(mode & 0o777 or 0o644)
            if destination.is_symlink():
                raise ValueError("Refusing to replace a symlinked plugin directory")
            if catalog_bytes(catalog_path) != original_catalog:
                raise ValueError("The marketplace changed during installation; retry without concurrent catalog edits")
            if destination.exists():
                old_manifest = json.loads((destination / "plugin.json").read_text())
                if old_manifest.get("name") != manifest["name"]:
                    raise ValueError("The destination belongs to another plugin")
                destination.rename(backup)
            try:
                stage.rename(destination)
                catalog_path.parent.mkdir(parents=True, exist_ok=True)
                if catalog_path.exists():
                    shutil.copy2(catalog_path, catalog_backup)
                temporary_catalog = catalog_path.with_suffix(".json.speech-tmp")
                temporary_catalog.write_text(json.dumps(catalog, indent=2) + "\n")
                if catalog_bytes(catalog_path) != original_catalog:
                    temporary_catalog.unlink()
                    raise ValueError("The marketplace changed during installation; retry without concurrent catalog edits")
                os.replace(temporary_catalog, catalog_path)
            except (OSError, ValueError):
                if destination.exists():
                    shutil.rmtree(destination)
                if backup.exists():
                    backup.rename(destination)
                raise
    print(json.dumps({
        "plugin_source": str(destination), "marketplace": str(catalog_path),
        "state": "Registered local source; restart the desktop app and install from Soniqo Local",
        "host_installation_verified": False,
    }, indent=2))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        install(args.archive.resolve(), Path.home())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, str(error) + "\n")
