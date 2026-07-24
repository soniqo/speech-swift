#!/usr/bin/env python3
"""Convert a local FLEURS split to the speech-swift ASR benchmark format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a <wav path>\\t<reference text> manifest from a local "
            "FLEURS language directory."
        )
    )
    parser.add_argument(
        "--fleurs-dir",
        type=Path,
        required=True,
        help="Language directory containing test.tsv and audio/test/.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--limit", type=int, help="Maximum number of rows.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_manifest(
    fleurs_dir: Path,
    split: str,
    output: Path,
    limit: int | None,
) -> int:
    source = fleurs_dir / f"{split}.tsv"
    audio_dir = fleurs_dir / "audio" / split
    if not source.is_file():
        raise FileNotFoundError(f"FLEURS metadata is missing: {source}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"FLEURS audio is missing: {audio_dir}")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")

    rows: list[str] = []
    with source.open(encoding="utf-8", newline="") as handle:
        for line_number, columns in enumerate(
            csv.reader(handle, delimiter="\t"),
            start=1,
        ):
            if len(columns) < 3:
                raise ValueError(
                    f"{source}:{line_number} has fewer than three columns"
                )
            audio = (audio_dir / columns[1]).resolve()
            if not audio.is_file():
                raise FileNotFoundError(
                    f"FLEURS audio referenced at line {line_number} "
                    f"is missing: {audio}"
                )
            transcript = columns[2].replace("\t", " ").strip()
            rows.append(f"{audio}\t{transcript}\n")
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"no usable rows in {source}")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(rows), encoding="utf-8")
    return len(rows)


def main() -> None:
    args = parse_args()
    count = build_manifest(
        args.fleurs_dir.expanduser().resolve(),
        args.split,
        args.output,
        args.limit,
    )
    print(f"Wrote {count} utterances to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
