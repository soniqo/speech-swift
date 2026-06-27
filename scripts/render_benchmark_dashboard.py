#!/usr/bin/env python3
"""Render the benchmark dashboard from persisted JSON reports."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Run:
    key: str
    path: Path
    metadata: dict[str, Any]
    asr: dict[str, Any] | None
    vad: dict[str, Any] | None
    diarization: dict[str, Any] | None

    @property
    def started_at(self) -> str:
        return str(self.metadata.get("started_at") or self.key)

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.started_at, self.key)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_runs(runs_dir: Path) -> list[Run]:
    runs: list[Run] = []
    if not runs_dir.exists():
        return runs

    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        metadata = load_json(run_path / "metadata.json")
        if not metadata:
            continue
        runs.append(Run(
            key=run_path.name,
            path=run_path,
            metadata=metadata,
            asr=load_json(run_path / "asr.json"),
            vad=load_json(run_path / "vad.json"),
            diarization=load_json(run_path / "diarization.json"),
        ))

    return sorted(runs, key=lambda run: run.sort_key, reverse=True)


def prune_runs(runs: list[Run], keep: int) -> None:
    if keep <= 0:
        return
    for run in runs[keep:]:
        shutil.rmtree(run.path, ignore_errors=True)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}{suffix}"


def mb(value: Any) -> str:
    try:
        return f"{float(value) / (1024 * 1024):.0f} MB"
    except (TypeError, ValueError):
        return "-"


def delta(current: Any, previous: Any, digits: int = 2, lower_is_better: bool = True) -> str:
    if current is None or previous is None:
        return '<span class="muted">new</span>'
    try:
        diff = float(current) - float(previous)
    except (TypeError, ValueError):
        return '<span class="muted">new</span>'
    if abs(diff) < 10 ** (-(digits + 1)):
        return '<span class="flat">0</span>'
    improved = diff < 0 if lower_is_better else diff > 0
    klass = "good" if improved else "bad"
    sign = "+" if diff > 0 else ""
    return f'<span class="{klass}">{sign}{diff:.{digits}f}</span>'


def asr_results(run: Run) -> list[dict[str, Any]]:
    if not run.asr:
        return []
    results = run.asr.get("results")
    return results if isinstance(results, list) else []


def vad_results(run: Run) -> list[dict[str, Any]]:
    if not run.vad:
        return []
    results = run.vad.get("results")
    return results if isinstance(results, list) else []


def diarization_results(run: Run) -> list[dict[str, Any]]:
    if not run.diarization:
        return []
    results = run.diarization.get("results")
    return results if isinstance(results, list) else []


def first_metric(result: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = result.get(key)
        if value is not None:
            return value
    return None


def previous_by_engine(runs: list[Run], latest: Run) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run.key == latest.key:
            continue
        for result in asr_results(run):
            engine = result.get("engine")
            if isinstance(engine, str) and engine not in out:
                out[engine] = result
    return out


def engine_history(runs: list[Run], engine: str, *metrics: str) -> list[float]:
    values: list[float] = []
    for run in reversed(runs):
        for result in asr_results(run):
            if result.get("engine") != engine:
                continue
            try:
                value = float(first_metric(result, *metrics))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return values


def sparkline(values: list[float], lower_is_better: bool) -> str:
    if len(values) < 2:
        return '<span class="muted">-</span>'
    values = values[-30:]
    width = 132
    height = 30
    min_v = min(values)
    max_v = max(values)
    spread = max(max_v - min_v, 1e-9)
    points: list[str] = []
    for idx, value in enumerate(values):
        x = (idx / max(len(values) - 1, 1)) * width
        normalized = (value - min_v) / spread
        y = height - (normalized * (height - 4)) - 2
        points.append(f"{x:.1f},{y:.1f}")
    trend_good = values[-1] <= values[0] if lower_is_better else values[-1] >= values[0]
    klass = "spark-good" if trend_good else "spark-bad"
    return (
        f'<svg class="spark {klass}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="trend"><polyline points="{" ".join(points)}"/></svg>'
    )


def latest_table(runs: list[Run]) -> str:
    if not runs:
        return '<p class="empty">No benchmark runs found.</p>'

    latest = runs[0]
    previous = previous_by_engine(runs, latest)
    rows: list[str] = []
    for result in sorted(asr_results(latest), key=lambda item: str(item.get("engine", ""))):
        engine = str(result.get("engine", "unknown"))
        prev = previous.get(engine, {})
        wer_agg = first_metric(result, "werAggregatePercent", "werPercent")
        prev_wer_agg = first_metric(prev, "werAggregatePercent", "werPercent")
        xrt_overall = first_metric(result, "throughputOverallXRT", "throughputXRT")
        prev_xrt_overall = first_metric(prev, "throughputOverallXRT", "throughputXRT")
        rows.append(
            "<tr>"
            f"<td><strong>{esc(engine)}</strong></td>"
            f"<td>{fmt(wer_agg, 2, '%')} {delta(wer_agg, prev_wer_agg, 2, True)}</td>"
            f"<td>{fmt(result.get('werMeanPercent'), 2, '%')}</td>"
            f"<td>{fmt(first_metric(result, 'cerAggregatePercent', 'cerPercent'), 2, '%')}</td>"
            f"<td>{fmt(xrt_overall, 1, 'x')} {delta(xrt_overall, prev_xrt_overall, 1, False)}</td>"
            f"<td>{fmt(result.get('throughputMedianXRT'), 1, 'x')}</td>"
            f"<td>{fmt(result.get('loadElapsedSeconds'), 1, 's')}</td>"
            f"<td>{mb(result.get('peakRSSBytes'))}</td>"
            f"<td>{int(result.get('utterances') or 0)}</td>"
            f"<td>{sparkline(engine_history(runs, engine, 'werAggregatePercent', 'werPercent'), True)}</td>"
            f"<td>{sparkline(engine_history(runs, engine, 'throughputOverallXRT', 'throughputXRT'), False)}</td>"
            "</tr>"
        )

    if not rows:
        return '<p class="empty">Latest run does not contain ASR results.</p>'

    return (
        '<table><thead><tr>'
        '<th>Engine</th><th>Agg WER</th><th>Avg WER</th><th>Agg CER</th>'
        '<th>Overall xRT</th><th>Median xRT</th><th>Load</th>'
        '<th>Peak RSS</th><th>Utts</th><th>WER trend</th><th>xRT trend</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def latest_vad_table(runs: list[Run]) -> str:
    latest = next((run for run in runs if vad_results(run)), None)
    if not latest:
        return '<p class="empty">No VAD benchmark results found.</p>'

    rows: list[str] = []
    for result in sorted(vad_results(latest), key=lambda item: str(item.get("engine", ""))):
        rows.append(
            "<tr>"
            f"<td><strong>{esc(result.get('engine', 'unknown'))}</strong></td>"
            f"<td>{fmt(result.get('fileF1Percent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('f1Percent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('precisionPercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('recallPercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('falseAlarmRatePercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('missRatePercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('throughputOverallXRT'), 1, 'x')}</td>"
            f"<td>{fmt(result.get('loadElapsedSeconds'), 1, 's')}</td>"
            f"<td>{mb(result.get('peakRSSBytes'))}</td>"
            f"<td>{int(result.get('files') or 0)}</td>"
            "</tr>"
        )

    return (
        '<table><thead><tr>'
        '<th>Engine</th><th>File F1</th><th>Span F1</th><th>Precision</th><th>Recall</th>'
        '<th>FAR</th><th>MR</th><th>Overall xRT</th><th>Load</th>'
        '<th>Peak RSS</th><th>Files</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def latest_diarization_table(runs: list[Run]) -> str:
    latest = next((run for run in runs if diarization_results(run)), None)
    if not latest:
        return '<p class="empty">No diarization benchmark results found.</p>'

    rows: list[str] = []
    for result in sorted(diarization_results(latest), key=lambda item: str(item.get("engine", ""))):
        rows.append(
            "<tr>"
            f"<td><strong>{esc(result.get('engine', 'unknown'))}</strong></td>"
            f"<td>{fmt(result.get('derPercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('jerMeanPercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('missPercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('falseAlarmPercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('speakerErrorPercent'), 2, '%')}</td>"
            f"<td>{fmt(result.get('speakerCountAccuracyPercent'), 1, '%')}</td>"
            f"<td>{fmt(result.get('throughputOverallXRT'), 1, 'x')}</td>"
            f"<td>{fmt(result.get('loadElapsedSeconds'), 1, 's')}</td>"
            f"<td>{mb(result.get('peakRSSBytes'))}</td>"
            f"<td>{int(result.get('files') or 0)}</td>"
            "</tr>"
        )

    return (
        '<table><thead><tr>'
        '<th>Engine</th><th>DER</th><th>JER</th><th>Miss</th><th>FA</th>'
        '<th>SE</th><th>Spk Acc</th><th>Overall xRT</th><th>Load</th>'
        '<th>Peak RSS</th><th>Files</th>'
        '</tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def run_history_table(runs: list[Run]) -> str:
    rows: list[str] = []
    for run in runs[:30]:
        metadata = run.metadata
        git = metadata.get("git") if isinstance(metadata.get("git"), dict) else {}
        runner = metadata.get("runner") if isinstance(metadata.get("runner"), dict) else {}
        results = asr_results(run)
        vad_count = len(vad_results(run))
        diarization_count = len(diarization_results(run))
        best_wer = min(
            (float(first_metric(r, "werAggregatePercent", "werPercent")) for r in results
             if first_metric(r, "werAggregatePercent", "werPercent") is not None),
            default=None,
        )
        fastest = max(
            (float(first_metric(r, "throughputOverallXRT", "throughputXRT")) for r in results
             if first_metric(r, "throughputOverallXRT", "throughputXRT") is not None),
            default=None,
        )
        rows.append(
            "<tr>"
            f"<td>{esc(run.started_at)}</td>"
            f"<td><code>{esc(git.get('short_sha', ''))}</code></td>"
            f"<td>{esc(runner.get('host') or runner.get('name') or '')}</td>"
            f"<td>{len(results)}</td>"
            f"<td>{vad_count}</td>"
            f"<td>{diarization_count}</td>"
            f"<td>{fmt(best_wer, 2, '%')}</td>"
            f"<td>{fmt(fastest, 1, 'x')}</td>"
            "</tr>"
        )
    return (
        '<table><thead><tr><th>Run</th><th>SHA</th><th>Host</th>'
        '<th>ASR</th><th>VAD</th><th>Diar</th><th>Best Agg WER</th><th>Fastest Overall xRT</th></tr></thead><tbody>'
        + "\n".join(rows)
        + '</tbody></table>'
    )


def render(runs: list[Run]) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    latest = runs[0] if runs else None
    metadata = latest.metadata if latest else {}
    runner = metadata.get("runner") if isinstance(metadata.get("runner"), dict) else {}
    git = metadata.get("git") if isinstance(metadata.get("git"), dict) else {}
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    latest_started = latest.started_at if latest else "-"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Soniqo Benchmarks</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0d1117;
      --panel: #161b22;
      --line: #30363d;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #2f81f7;
      --good: #3fb950;
      --bad: #f85149;
      --flat: #d29922;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0; font-size: 28px; line-height: 1.1; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    p {{ margin: 0; }}
    .muted {{ color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 16px 0 24px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      min-width: 0;
    }}
    .label {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .value {{ font-weight: 650; overflow-wrap: anywhere; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .good {{ color: var(--good); margin-left: 6px; }}
    .bad {{ color: var(--bad); margin-left: 6px; }}
    .flat {{ color: var(--flat); margin-left: 6px; }}
    .spark {{ display: block; width: 132px; height: 30px; }}
    .spark polyline {{ fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }}
    .spark-good polyline {{ stroke: var(--good); }}
    .spark-bad polyline {{ stroke: var(--bad); }}
    .empty {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: var(--panel);
    }}
    @media (max-width: 840px) {{
      header {{ display: block; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Soniqo Benchmarks</h1>
        <p class="muted">Generated {esc(generated)}</p>
      </div>
      <p class="muted">Local ASR quality and speed on Apple Silicon</p>
    </header>

    <section class="summary" aria-label="Latest run summary">
      <div class="stat"><div class="label">Latest run</div><div class="value">{esc(latest_started)}</div></div>
      <div class="stat"><div class="label">Commit</div><div class="value"><code>{esc(git.get("short_sha", "-"))}</code></div></div>
      <div class="stat"><div class="label">Runner</div><div class="value">{esc(runner.get("host") or runner.get("name") or "-")}</div></div>
      <div class="stat"><div class="label">Dataset</div><div class="value">{esc(config.get("dataset", "-"))}, n={esc(config.get("asr_limit", "-"))}</div></div>
    </section>

    <h2>Latest ASR Results</h2>
    {latest_table(runs)}

    <h2>Latest VAD Results</h2>
    {latest_vad_table(runs)}

    <h2>Latest Diarization Results</h2>
    {latest_diarization_table(runs)}

    <h2>Run History</h2>
    {run_history_table(runs) if runs else '<p class="empty">No run history yet.</p>'}
  </main>
</body>
</html>
"""


def write_summary(runs: list[Run], path: Path) -> None:
    latest = runs[0] if runs else None
    with path.open("a", encoding="utf-8") as f:
        f.write("## Benchmark Summary\n\n")
        if not latest:
            f.write("No benchmark runs found.\n")
            return
        f.write(f"Latest run: `{latest.started_at}`\n\n")
        if vad_results(latest):
            f.write("VAD and diarization sections are available in the HTML dashboard when their JSON files exist.\n\n")
        f.write("| Engine | Agg WER | Avg WER | Agg CER | Overall xRT | Median xRT | Load | Peak RSS |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for result in sorted(asr_results(latest), key=lambda item: str(item.get("engine", ""))):
            f.write(
                f"| {result.get('engine', 'unknown')} "
                f"| {fmt(first_metric(result, 'werAggregatePercent', 'werPercent'), 2, '%')} "
                f"| {fmt(result.get('werMeanPercent'), 2, '%')} "
                f"| {fmt(first_metric(result, 'cerAggregatePercent', 'cerPercent'), 2, '%')} "
                f"| {fmt(first_metric(result, 'throughputOverallXRT', 'throughputXRT'), 1, 'x')} "
                f"| {fmt(result.get('throughputMedianXRT'), 1, 'x')} "
                f"| {fmt(result.get('loadElapsedSeconds'), 1, 's')} "
                f"| {mb(result.get('peakRSSBytes'))} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--prune", type=int, default=0, help="Keep only the newest N run directories")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if args.prune:
        prune_runs(runs, args.prune)
        runs = load_runs(args.runs_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(runs), encoding="utf-8")

    if args.summary:
        write_summary(runs, args.summary)


if __name__ == "__main__":
    main()
