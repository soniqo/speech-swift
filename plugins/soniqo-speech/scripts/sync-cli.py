#!/usr/bin/env python3
"""Freeze each CLI command and its argument contract for the plugin release."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def catalogue(binary, version):
    document = json.loads(subprocess.check_output(
        [str(binary), "--experimental-dump-help"], timeout=30))
    if document.get("serializationVersion") != 0:
        raise ValueError("Unsupported CLI help format")
    commands = []
    def append_command(command, parent=()):
        path = (*parent, command["commandName"])
        arguments = []
        for arg in command.get("arguments", []):
            if arg["valueName"] == "help":
                continue
            preferred = arg.get("preferredName")
            flag = None
            if preferred:
                flag = ("--" if preferred["kind"] == "long" else "-") + preferred["name"]
            default = arg.get("defaultValue")
            scalar = "string"
            if arg["kind"] == "flag":
                scalar = "boolean"
            elif default and re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", default):
                scalar = "number"
            arguments.append({
                "key": arg["valueName"].replace("-", "_"),
                "kind": arg["kind"], "flag": flag,
                "required": not arg["isOptional"], "repeating": arg["isRepeating"],
                "type": scalar, "description": arg.get("abstract", ""),
                **({"default": default} if default is not None else {}),
                **({"choices": arg["allValues"]} if "allValues" in arg else {}),
            })
        keys = [arg["key"] for arg in arguments]
        if len(keys) != len(set(keys)) or "execution" in keys:
            raise ValueError("Ambiguous CLI arguments for " + " ".join(path))
        children = command.get("subcommands", [])
        commands.append({
            "command": " ".join(path), "path": list(path),
            "tool": "speech_" + "_".join(path).replace("-", "_"),
            "description": command.get("abstract", "Show speech command help."),
            "arguments": arguments,
            **({"subcommands": [" ".join((*path, child["commandName"])) for child in children]} if children else {}),
        })
        for child in children:
            append_command(child, path)
    for command in document["command"]["subcommands"]:
        append_command(command)
    return {
        "runtime_version": version,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "help_sha256": hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
        "commands": commands,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--runtime-version", required=True)
    args = parser.parse_args()
    data = catalogue(args.binary.resolve(), args.runtime_version)
    (ROOT / "src/cli-catalog.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"commands": len(data["commands"]),
                      "arguments": sum(len(c["arguments"]) for c in data["commands"]),
                      "runtime_version": data["runtime_version"]}))
