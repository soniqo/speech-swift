#!/usr/bin/env python3
"""Build and inspect the desktop speech package without publishing it."""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
URL_FIELDS = ("websiteURL", "supportURL", "privacyPolicyURL", "termsOfServiceURL")
RUNTIME_BUNDLES = frozenset({
    "Qwen3Speech_KokoroTTS.bundle", "Qwen3Speech_MagpieTTS.bundle",
    "Qwen3Speech_VoiceChat.bundle", "swift-crypto_Crypto.bundle",
    "swift-nio_NIOPosix.bundle", "swift-transformers_Hub.bundle",
    "Qwen3Speech_Clef.bundle", "Qwen3Speech_GLiNER.bundle", "Qwen3Speech_VoxtralTTS.bundle",
})


def check_regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Package inputs must be regular files, not symlinks: " + path.name)


def copy_regular(source, destination):
    check_regular(source)
    shutil.copy2(source, destination, follow_symlinks=False)
    check_regular(destination)


def copy_tree(source, destination):
    if not stat.S_ISDIR(source.lstat().st_mode):
        raise ValueError("Package directories must not be symlinks: " + source.name)
    for path in source.rglob("*"):
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            raise ValueError("Package trees must not contain symlinks or special files: " + path.name)
    shutil.copytree(source, destination, symlinks=True)


def validate(files, submission=False):
    manifest = json.loads(files["plugin.json"])
    mcp = json.loads(files["mcp.json"])
    if manifest.get("$schema") != "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json":
        raise ValueError("Incorrect portable manifest schema")
    if mcp.get("$schema") != "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json":
        raise ValueError("Incorrect MCP schema")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", manifest["name"]):
        raise ValueError("Invalid plugin identity")
    if not re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"]):
        raise ValueError("Invalid plugin version")
    extension = manifest["extensions"]["com.openai"]
    interface = extension["interface"]
    if manifest.get("apps") or extension.get("apps") or ".app.json" in files:
        raise ValueError("Public packages must not contain private app bindings")
    if "hooks/hooks.json" in files or extension.get("hooks"):
        raise ValueError("Public packages must not contain lifecycle hooks")
    if not 1 <= len(interface["displayName"]) <= 30 or not 1 <= len(interface["shortDescription"]) <= 30:
        raise ValueError("Listing name and subtitle must be at most 30 characters")
    if interface["category"] != "Productivity":
        raise ValueError("Unexpected listing category")
    prompts = interface["defaultPrompt"]
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 3:
        raise ValueError("Supply one to three starter prompts")
    if any(not p.strip() or len(p) > 128 or "\n" in p for p in prompts):
        raise ValueError("Invalid starter prompt")
    for key in ("logo", "composerIcon", "logoDark", "composerIconDark"):
        if key not in interface:
            continue
        path = interface[key].removeprefix("./")
        if path not in files:
            raise ValueError("Missing icon: " + path)
        data = files[path]
        if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) > 5 * 1024 * 1024:
            raise ValueError("Icons must be PNGs no larger than 5 MiB")
        width, height = struct.unpack(">II", data[16:24])
        minimum = 256 if key.startswith("logo") else 48
        if width != height or width < minimum:
            raise ValueError("Icon is not square or is too small: " + path)
    if mcp["mcpServers"] != {"speech": {
            "type": "stdio", "command": "./scripts/start.sh", "args": [], "cwd": "${PLUGIN_ROOT}"}}:
        raise ValueError("Unexpected desktop server transport or launch command")
    skill = files["skills/speech/SKILL.md"].decode()
    if not skill.startswith("---\nname: speech\n") or "\ndescription: " not in skill:
        raise ValueError("Invalid skill frontmatter")
    if "dist/cli-catalog.json" not in files:
        raise ValueError("The frozen CLI catalogue is missing")
    catalog = json.loads(files["dist/cli-catalog.json"])
    if not catalog.get("commands") or len({c["tool"] for c in catalog["commands"]}) != len(catalog["commands"]):
        raise ValueError("The CLI catalogue has missing or duplicate command tools")
    if "native/speech" in files and hashlib.sha256(files["native/speech"]).hexdigest() != catalog["binary_sha256"]:
        raise ValueError("The packaged CLI binary does not match the catalogue")
    gaps = []
    for field in URL_FIELDS:
        value = interface.get(field)
        if not value:
            gaps.append("Missing verified listing URL: " + field)
        elif not value.startswith("https://"):
            raise ValueError("Listing URLs must use HTTPS")
    review = extension.get("review", {})
    cases = review.get("test_cases", {})
    if len(cases.get("positive", [])) != 5 or len(cases.get("negative", [])) != 3:
        gaps.append("Initial MCP review requires five positive and three negative cases")
    if not review.get("demo_recording_url"):
        gaps.append("Missing verified recording of the plugin in the desktop host")
    if "native/speech" not in files:
        gaps.append("Native runtime not bundled; this local package requires brew install speech")
    if submission and gaps:
        raise ValueError("Not ready for public submission:\n- " + "\n- ".join(gaps))
    return manifest, gaps


def add_native_runtime(stage, source, version):
    binary = source / "speech"
    check_regular(binary)
    check_regular(source / "mlx.metallib")
    bundles = {path.name for path in source.iterdir() if path.name.endswith(".bundle")}
    if bundles != RUNTIME_BUNDLES:
        raise ValueError("Runtime resource bundles differ from the reviewed allowlist; review this release's resources first")
    description = subprocess.check_output(["file", "-b", str(binary)], text=True)
    if "Mach-O" not in description or "arm64" not in description:
        raise ValueError("The bundled runtime must be a native Apple Silicon executable")
    dependencies = subprocess.check_output(["otool", "-L", str(binary)], text=True).splitlines()[1:]
    for dependency in dependencies:
        path = dependency.strip().split(" ", 1)[0]
        if not path.startswith(("/System/Library/", "/usr/lib/")):
            raise ValueError("The runtime has a non-system dependency: " + path)
    snapshot = json.loads((ROOT / "src/cli-catalog.json").read_text())
    if snapshot["runtime_version"] != version or hashlib.sha256(binary.read_bytes()).hexdigest() != snapshot["binary_sha256"]:
        raise ValueError("The CLI catalogue differs from this runtime; run scripts/sync-cli.py first")
    spec = importlib.util.spec_from_file_location("speech_cli_catalog", ROOT / "scripts/sync-cli.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    actual = generator.catalogue(binary, version)
    if actual != snapshot:
        raise ValueError("CLI command or argument definitions differ from the reviewed catalogue")
    destination = stage / "native"
    destination.mkdir()
    copy_regular(binary, destination / "speech")
    for name in sorted(RUNTIME_BUNDLES):
        copy_tree(source / name, destination / name)
    copy_regular(source / "mlx.metallib", destination / "mlx.metallib")
    (destination / "runtime.json").write_text(json.dumps({
        "version": version,
        "platform": "macOS 15+ arm64",
        "cli_sha256": snapshot["binary_sha256"],
        "help_sha256": actual["help_sha256"],
        "commands": len(snapshot["commands"]),
        "arguments": sum(len(command["arguments"]) for command in snapshot["commands"]),
        "source": "speech runtime release",
    }, indent=2) + "\n")


def package(output, runtime=None, runtime_version=None, submission=False):
    with tempfile.TemporaryDirectory(prefix="soniqo-speech-package-") as temporary:
        stage = Path(temporary) / "soniqo-speech"
        stage.mkdir()
        for filename in ("plugin.json", "mcp.json"):
            copy_regular(ROOT / filename, stage / filename)
        for folder in ("dist", "assets", "skills", "review"):
            source = ROOT / folder
            if source.exists():
                copy_tree(source, stage / folder)
        (stage / "scripts").mkdir()
        copy_regular(ROOT / "scripts/start.sh", stage / "scripts/start.sh")
        if not (stage / "scripts/start.sh").stat().st_mode & 0o111:
            raise ValueError("The contained desktop launcher must be executable")
        copy_regular(REPO / "LICENSE", stage / "LICENSE")
        if not (stage / "dist/server.cjs").is_file():
            raise ValueError("Build the server before packaging")
        if (stage / "dist/cli-catalog.json").read_bytes() != (ROOT / "src/cli-catalog.json").read_bytes():
            raise ValueError("Rebuild the server after updating the CLI catalogue")
        fixture = stage / "review/fixtures/known-speech.wav"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        copy_regular(REPO / "Tests/SpeechPluginTests/Fixtures/known-speech.wav", fixture)
        if runtime:
            if not runtime_version:
                raise ValueError("Provide the bundled runtime version explicitly")
            add_native_runtime(stage, runtime, runtime_version)
        files = {}
        for path in sorted(stage.rglob("*")):
            if path.is_symlink():
                raise ValueError("Packages must not contain symlinks")
            if path.is_file():
                files[path.relative_to(stage).as_posix()] = path.read_bytes()
        manifest, gaps = validate(files, submission)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_zip = output.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temporary_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in files:
                archive.write(stage / name, name)
        with zipfile.ZipFile(temporary_zip) as archive:
            if archive.testzip():
                raise ValueError("Archive integrity check failed")
            archived = {name: archive.read(name) for name in archive.namelist()}
            validate(archived, submission)
            for info in archive.infolist():
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Archive contains a symlink")
        temporary_zip.replace(output)
        report = {
            "name": manifest["name"], "version": manifest["version"],
            "archive": str(output), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "files": sorted(files), "public_submission_ready": False,
            "remaining": gaps + [
                "Confirm the selected developer identity is verified",
                "Confirm organization eligibility for local MCP public submissions",
                "Complete the desktop-host evaluation and required portal checks",
            ],
        }
        output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-runtime", type=Path)
    parser.add_argument("--runtime-version")
    parser.add_argument("--submission", action="store_true", help="Reject a package missing public review materials")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "plugin.json").read_text())
    target = args.output or REPO / ".build/plugins" / ("soniqo-speech-" + manifest["version"] + "-local.zip")
    try:
        package(target.resolve(), args.native_runtime, args.runtime_version, args.submission)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + "\n")
