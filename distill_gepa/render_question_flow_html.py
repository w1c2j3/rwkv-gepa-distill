from __future__ import annotations

import argparse
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import orjson

from .trajectory_schema import parse_slot_id
from .world_schema import BenchmarkQuestion, load_benchmark_question_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one-question HTML flow report.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--question-id", type=str, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield payload


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    payload = orjson.loads(path.read_bytes())
    return payload if isinstance(payload, dict) else {}


def html_escape(value: Any) -> str:
    return html.escape(str(value))


def short_text(value: str, *, limit: int = 120) -> str:
    text = " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def select_question_id(
    *,
    questions_by_id: dict[str, BenchmarkQuestion],
    decisions_by_id: dict[str, dict[str, Any]],
    gepa_counts: Counter[str],
    final_counts: Counter[str],
) -> str:
    ranked_ids = sorted(
        questions_by_id,
        key=lambda question_id: (
            int(decisions_by_id.get(question_id, {}).get("classification") == "needs_optimization"),
            int(gepa_counts[question_id] > 0),
            final_counts[question_id],
            question_id,
        ),
        reverse=True,
    )
    if not ranked_ids:
        raise ValueError("No questions available for HTML rendering")
    return ranked_ids[0]


def source_badge(source_type: str) -> str:
    if source_type == "gepa_complex":
        return '<span class="badge badge-gepa">GEPA</span>'
    if source_type == "benchmark_complex":
        return '<span class="badge badge-benchmark">Benchmark</span>'
    return f'<span class="badge badge-muted">{html_escape(source_type)}</span>'


def render_metric_cards(items: list[tuple[str, Any]]) -> str:
    return "".join(
        f"""
        <div class="metric-card">
          <div class="metric-label">{html_escape(label)}</div>
          <div class="metric-value">{html_escape(value)}</div>
        </div>
        """
        for label, value in items
    )


def render_options(question: BenchmarkQuestion) -> str:
    if not question.choices:
        return '<div class="muted">Open QA</div>'
    return "".join(
        f'<li><span class="option-label">{chr(65 + index)}.</span> {html_escape(choice)}</li>'
        for index, choice in enumerate(question.choices)
    )


def render_benchmark_matrix(
    *,
    models: list[str],
    sample_indices: list[int],
    benchmark_rows: list[dict[str, Any]],
) -> str:
    rows_by_key = {(row["model_name"], row["sample_index"]): row for row in benchmark_rows}
    header = "".join(f"<th>S{sample_index}</th>" for sample_index in sample_indices)
    body_rows: list[str] = []
    for model_name in models:
        cells: list[str] = []
        for sample_index in sample_indices:
            row = rows_by_key.get((model_name, sample_index))
            if row is None:
                cells.append('<td class="cell-empty">-</td>')
                continue
            is_correct = row.get("correct") is True
            classes = "cell-ok" if is_correct else "cell-bad"
            tooltip = f"perm={row.get('choice_permutation')} | {short_text(str(row.get('assistant', '')), limit=110)}"
            cells.append(f'<td class="{classes}" title="{html_escape(tooltip)}">{"✓" if is_correct else "✗"}</td>')
        body_rows.append(f"<tr><th>{html_escape(model_name)}</th>{''.join(cells)}</tr>")
    return f"""
    <table class="matrix-table">
      <thead>
        <tr><th>Model</th>{header}</tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
    """


def render_complex_matrix(
    *,
    models: list[str],
    sample_indices: list[int],
    complex_rows: list[dict[str, Any]],
) -> str:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in complex_rows:
        slot = parse_slot_id(row["meta"]["slot_id"])
        rows_by_key[(slot.target_model, slot.sample_index)] = row
    header = "".join(f"<th>S{sample_index}</th>" for sample_index in sample_indices)
    body_rows: list[str] = []
    for model_name in models:
        cells: list[str] = []
        for sample_index in sample_indices:
            row = rows_by_key.get((model_name, sample_index))
            if row is None:
                cells.append('<td class="cell-empty">-</td>')
                continue
            source_type = str(row.get("source_type"))
            classes = "cell-gepa" if source_type == "gepa_complex" else "cell-benchmark"
            label = "G" if source_type == "gepa_complex" else "B"
            cells.append(f'<td class="{classes}" title="{html_escape(source_type)}">{label}</td>')
        body_rows.append(f"<tr><th>{html_escape(model_name)}</th>{''.join(cells)}</tr>")
    return f"""
    <table class="matrix-table">
      <thead>
        <tr><th>Retained Complex</th>{header}</tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
    """


def render_complex_empty_slots(
    *,
    models: list[str],
    sample_indices: list[int],
    benchmark_rows: list[dict[str, Any]],
    complex_rows: list[dict[str, Any]],
    gepa_rows: list[dict[str, Any]],
) -> str:
    benchmark_by_key = {(row["model_name"], row["sample_index"]): row for row in benchmark_rows}
    complex_keys = {
        (slot.target_model, slot.sample_index)
        for row in complex_rows
        for slot in [parse_slot_id(row["meta"]["slot_id"])]
    }
    gepa_models = {str(row.get("target_model")) for row in gepa_rows if isinstance(row.get("target_model"), str)}
    items: list[str] = []
    for model_name in models:
        for sample_index in sample_indices:
            key = (model_name, sample_index)
            if key in complex_keys:
                continue
            benchmark_row = benchmark_by_key.get(key)
            if benchmark_row is None:
                reason = "slot missing from benchmark output"
            elif benchmark_row.get("correct") is True:
                reason = "not empty in retention, check render ordering"
            elif model_name in gepa_models:
                reason = "benchmark wrong and GEPA did not recover this slot"
            else:
                reason = "benchmark wrong and this model-slot never produced a correct complex trajectory"
            items.append(
                f"<li><strong>{html_escape(model_name)} · S{sample_index}</strong> <span class=\"muted\">{html_escape(reason)}</span></li>"
            )
    if not items:
        return '<div class="empty-card" style="padding:16px 18px;">No empty retention slots for this question.</div>'
    return f'<ul class="reason-list">{"".join(items)}</ul>'


def render_gepa_cards(gepa_rows: list[dict[str, Any]]) -> str:
    if not gepa_rows:
        return '<div class="empty-card">This question did not enter GEPA.</div>'
    cards: list[str] = []
    for row in gepa_rows:
        delta = row["question_delta"]
        improved = bool(delta.get("improved_to_stable_correct"))
        cards.append(
            f"""
            <div class="info-card">
              <div class="card-header">
                <span class="card-title">{html_escape(str(row.get("target_model")))}</span>
                {'<span class="badge badge-hit">Stable</span>' if improved else '<span class="badge badge-muted">Not Stable</span>'}
              </div>
              <div class="card-grid">
                <div><span class="muted">Domain</span><strong>{html_escape(str(row.get("domain")))}</strong></div>
                <div><span class="muted">Baseline</span><strong>{delta.get("baseline_correct_rate")} / stable={delta.get("baseline_stable_correct")}</strong></div>
                <div><span class="muted">Optimized</span><strong>{delta.get("optimized_correct_rate")} / stable={delta.get("optimized_stable_correct")}</strong></div>
                <div><span class="muted">Run Dir</span><strong>{html_escape(str(row.get("run_dir")))}</strong></div>
              </div>
            </div>
            """
        )
    return "".join(cards)


def render_row_examples(title: str, rows: list[dict[str, Any]], *, limit: int = 3) -> str:
    if not rows:
        return ""
    cards: list[str] = []
    for row in rows[:limit]:
        meta = row.get("meta", {})
        cards.append(
            f"""
            <div class="example-card">
              <div class="example-top">
                {source_badge(str(row.get("source_type")))}
                <code>{html_escape(meta.get("trajectory_id", ""))}</code>
              </div>
              <div class="example-block">
                <div class="example-label">User</div>
                <pre>{html_escape(str(row.get("user", "")))}</pre>
              </div>
              <div class="example-block">
                <div class="example-label">Assistant</div>
                <pre>{html_escape(str(row.get("assistant", "")))}</pre>
              </div>
            </div>
            """
        )
    return f"""
    <div class="subsection">
      <h3>{html_escape(title)}</h3>
      <div class="example-grid">
        {''.join(cards)}
      </div>
    </div>
    """


def render_wrong_samples(rows: list[dict[str, Any]], *, limit: int = 6) -> str:
    if not rows:
        return '<div class="empty-card" style="padding:16px 18px;">No benchmark failures for this question.</div>'
    items = "".join(
        f'<div class="wrong-item"><strong>{html_escape(row["model_name"])} · S{row["sample_index"]}</strong><div class="muted">perm={html_escape(row.get("choice_permutation"))}</div><div style="margin-top:8px;">{html_escape(short_text(str(row.get("assistant", "")), limit=220))}</div></div>'
        for row in rows[:limit]
    )
    return f'<div class="wrong-list">{items}</div>'


def render_rewrite_groups(rewrite_rows: list[dict[str, Any]]) -> str:
    if not rewrite_rows:
        return '<div class="empty-card">No rewrite rows kept for this question.</div>'
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rewrite_rows:
        grouped[str(row["meta"]["parent_trajectory_id"])].append(row)
    cards: list[str] = []
    for parent_trajectory_id, rows in sorted(grouped.items())[:12]:
        prompt_items = "".join(
            f"<li>{html_escape(short_text(str(row.get('user', '')), limit=180))}</li>"
            for row in sorted(rows, key=lambda item: str(item["meta"]["trajectory_id"]))
        )
        cards.append(
            f"""
            <div class="rewrite-card">
              <div class="card-header">
                <span class="card-title">{html_escape(parent_trajectory_id)}</span>
                <span class="badge badge-rewrite">{len(rows)} rewrites</span>
              </div>
              <ul class="rewrite-list">{prompt_items}</ul>
            </div>
            """
        )
    hidden_count = max(0, len(grouped) - 12)
    hidden_note = f'<div class="muted">+ {hidden_count} more parent trajectories omitted from the gallery.</div>' if hidden_count else ""
    return f'<div class="rewrite-grid">{"".join(cards)}</div>{hidden_note}'


def build_html(
    *,
    dataset_dir: Path,
    summary: dict[str, Any],
    question: BenchmarkQuestion,
    decision: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    gepa_rows: list[dict[str, Any]],
    complex_rows: list[dict[str, Any]],
    rewrite_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> str:
    benchmark_storage = summary.get("benchmark_storage", {})
    models = benchmark_storage.get("models")
    if not isinstance(models, list) or not models:
        models = sorted({str(row["model_name"]) for row in benchmark_rows})
    sample_indices = sorted({int(row["sample_index"]) for row in benchmark_rows})
    if not sample_indices:
        sample_indices = list(range(8))

    benchmark_correct_count = sum(1 for row in benchmark_rows if row.get("correct") is True)
    complex_source_counts = Counter(str(row.get("source_type")) for row in complex_rows)
    rewrite_parent_count = len({str(row["meta"]["parent_trajectory_id"]) for row in rewrite_rows})
    per_model = decision.get("per_model", {}) if isinstance(decision, dict) else {}
    wrong_rows = [row for row in benchmark_rows if row.get("correct") is not True]

    model_summary_cards = render_metric_cards(
        [
            (
                model_name,
                f"{per_model.get(model_name, {}).get('correct_count', 0)}/{per_model.get(model_name, {}).get('sample_count', 0)}",
            )
            for model_name in models
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Question Flow Report</title>
  <style>
    :root {{
      --bg: #f5efe3;
      --panel: rgba(255,255,255,0.82);
      --ink: #1d1b19;
      --muted: #6f675d;
      --line: rgba(29,27,25,0.12);
      --gold: #c78522;
      --green: #207b52;
      --green-soft: #dcefe5;
      --red: #b04131;
      --red-soft: #f5ddd7;
      --blue: #2a5f92;
      --blue-soft: #dbe8f6;
      --shadow: 0 24px 60px rgba(55, 41, 22, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(199,133,34,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(42,95,146,0.12), transparent 30%),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    .page {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 40px 28px 72px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,248,234,0.9));
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 28px;
      padding: 30px 32px;
      margin-bottom: 28px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1, h2, h3 {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 0;
    }}
    h1 {{
      font-size: 36px;
      line-height: 1.08;
      margin-bottom: 12px;
    }}
    h2 {{
      font-size: 26px;
      margin-bottom: 16px;
    }}
    h3 {{
      font-size: 18px;
      margin-bottom: 12px;
    }}
    p {{
      margin: 0;
      line-height: 1.6;
    }}
    .hero-grid, .stats-grid, .section-grid, .card-grid, .example-grid, .rewrite-grid {{
      display: grid;
      gap: 16px;
    }}
    .hero-grid {{
      grid-template-columns: 1.25fr 0.75fr;
      margin-top: 24px;
    }}
    .stats-grid {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .metric-card, .section, .info-card, .example-card, .rewrite-card, .empty-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .metric-card {{
      padding: 18px 20px;
    }}
    .metric-label, .muted, .example-label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .metric-value {{
      font-size: 30px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .section {{
      padding: 24px 26px;
      margin-bottom: 18px;
    }}
    .section-head {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .step {{
      width: 36px;
      height: 36px;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #1d1b19, #6d5e4c);
      color: #fff;
      font-weight: 700;
    }}
    .question-box {{
      padding: 18px 20px;
      background: rgba(255,255,255,0.66);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .question-box ul {{
      margin: 16px 0 0;
      padding-left: 0;
      list-style: none;
    }}
    .question-box li {{
      padding: 8px 0;
      border-top: 1px dashed var(--line);
    }}
    .question-box li:first-child {{
      border-top: none;
    }}
    .option-label {{
      display: inline-block;
      width: 24px;
      color: var(--gold);
      font-weight: 700;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .badge-hit {{ background: #e0f3e9; color: var(--green); }}
    .badge-muted {{ background: #ece8e3; color: #746a60; }}
    .badge-benchmark {{ background: #f7ead2; color: #9a5d0f; }}
    .badge-gepa {{ background: var(--blue-soft); color: var(--blue); }}
    .badge-rewrite {{ background: #efe3ff; color: #6d41a9; }}
    .matrix-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 8px;
      table-layout: fixed;
    }}
    .matrix-table th, .matrix-table td {{
      text-align: center;
      padding: 10px 8px;
      border-radius: 14px;
      font-size: 13px;
    }}
    .matrix-table th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .cell-ok {{ background: var(--green-soft); color: var(--green); font-weight: 700; }}
    .cell-bad {{ background: var(--red-soft); color: var(--red); font-weight: 700; }}
    .cell-gepa {{ background: var(--blue-soft); color: var(--blue); font-weight: 700; }}
    .cell-benchmark {{ background: #f7ead2; color: #9a5d0f; font-weight: 700; }}
    .cell-empty {{ background: rgba(0,0,0,0.05); color: var(--muted); }}
    .info-card, .rewrite-card {{
      padding: 18px 20px;
    }}
    .card-header, .example-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}
    .card-title {{
      font-weight: 700;
      font-size: 16px;
    }}
    .card-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .card-grid div {{
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
    }}
    .card-grid strong {{
      display: block;
      margin-top: 4px;
      font-size: 14px;
    }}
    .example-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .example-card {{
      padding: 18px;
    }}
    .example-block + .example-block {{
      margin-top: 12px;
    }}
    pre, code {{
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    }}
    pre {{
      margin: 6px 0 0;
      padding: 12px 14px;
      border-radius: 14px;
      background: #161514;
      color: #f6f0e4;
      overflow-x: auto;
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 12px;
    }}
    .wrong-list {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .wrong-item {{
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}
    .rewrite-list {{
      margin: 0;
      padding-left: 18px;
    }}
    .legend-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .reason-list {{
      margin: 0;
      padding-left: 20px;
      display: grid;
      gap: 8px;
    }}
    .rewrite-list li + li {{
      margin-top: 8px;
    }}
    .subsection {{
      margin-top: 18px;
    }}
    @media (max-width: 900px) {{
      .hero-grid, .stats-grid, .example-grid, .wrong-list, .card-grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 24px 16px 48px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Single Question Flow</div>
      <h1>{html_escape(question.question_text)}</h1>
      <p>{html_escape(dataset_dir.name)} · {html_escape(question.domain)} · {html_escape(question.question_id)}</p>
      <div class="hero-grid">
        <div class="question-box">
          <h3>Canonical Question</h3>
          <p style="margin-top:10px;">{html_escape(question.question_text)}</p>
          <ul>{render_options(question)}</ul>
          <p style="margin-top:16px;"><strong>Gold Answer:</strong> {html_escape(question.gold_answer)}</p>
        </div>
        <div class="stats-grid">
          {render_metric_cards([
            ("Classification", decision.get("classification", "unknown")),
            ("Benchmark Correct", f"{benchmark_correct_count}/{len(benchmark_rows)}"),
            ("Complex Retained", len(complex_rows)),
            ("Final Rows", len(final_rows)),
          ])}
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="step">1</span><h2>Benchmark</h2></div>
      <div class="stats-grid">{model_summary_cards}</div>
      <div class="subsection">
        <h3>32-slot correctness matrix</h3>
        <p class="muted" style="margin-bottom:10px;">S0-S7 are fixed sample_index slots for each model on this question.</p>
        {render_benchmark_matrix(models=models, sample_indices=sample_indices, benchmark_rows=benchmark_rows)}
      </div>
      <div class="subsection">
        <h3>Failure samples</h3>
        {render_wrong_samples(wrong_rows)}
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="step">2</span><h2>Routing</h2></div>
      <p>The benchmark summary decides whether this question is direct distill or enters GEPA.</p>
      <div class="stats-grid" style="margin-top:16px;">
        {render_metric_cards([
          (model_name, f"stable={per_model.get(model_name, {}).get('stable_correct')} · gepa={per_model.get(model_name, {}).get('usable_for_gepa')}")
          for model_name in models
        ])}
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="step">3</span><h2>GEPA</h2></div>
      <p>GEPA works at the same-domain, target-model group level. Success only counts if the question becomes stable correct.</p>
      <div class="section-grid">{render_gepa_cards(gepa_rows)}</div>
    </section>

    <section class="section">
      <div class="section-head"><span class="step">4</span><h2>Complex Retention</h2></div>
      <div class="stats-grid">
        {render_metric_cards([
          ("Benchmark Complex", complex_source_counts.get("benchmark_complex", 0)),
          ("GEPA Complex", complex_source_counts.get("gepa_complex", 0)),
          ("Per-Question Complex Cap", "32"),
          ("Actual Complex Rows", len(complex_rows)),
        ])}
      </div>
      <div class="subsection">
        <h3>Which source won each slot</h3>
        <p class="muted" style="margin-bottom:10px;">Each cell shows the retained complex trajectory for one fixed slot. Empty means that slot had no correct complex row to keep.</p>
        {render_complex_matrix(models=models, sample_indices=sample_indices, complex_rows=complex_rows)}
        <div class="legend-row">
          <span class="badge badge-benchmark">B = benchmark correct kept</span>
          <span class="badge badge-gepa">G = GEPA correct kept</span>
          <span class="badge badge-muted">- = no correct complex trajectory kept for this slot</span>
        </div>
      </div>
      <div class="subsection">
        <h3>Why some slots are empty</h3>
        {render_complex_empty_slots(
          models=models,
          sample_indices=sample_indices,
          benchmark_rows=benchmark_rows,
          complex_rows=complex_rows,
          gepa_rows=gepa_rows,
        )}
      </div>
      {render_row_examples("Representative complex rows", complex_rows, limit=4)}
    </section>

    <section class="section">
      <div class="section-head"><span class="step">5</span><h2>Rewrite Expansion</h2></div>
      <div class="stats-grid">
        {render_metric_cards([
          ("Rewrite Rows", len(rewrite_rows)),
          ("Parent Complex Trajectories", rewrite_parent_count),
          ("Average Rewrites / Parent", round(len(rewrite_rows) / max(1, rewrite_parent_count), 2)),
          ("Per-Question Final Cap", "128"),
        ])}
      </div>
      <div class="subsection">
        <h3>Rewrite gallery</h3>
        {render_rewrite_groups(rewrite_rows)}
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="step">6</span><h2>Final Export</h2></div>
      <div class="stats-grid">
        {render_metric_cards([
          ("Complex Rows", len(complex_rows)),
          ("Rewrite Rows", len(rewrite_rows)),
          ("Final SFT Rows", len(final_rows)),
          ("Hit 128 Cap", "Yes" if len(final_rows) >= 128 else "No"),
        ])}
      </div>
      <p style="margin-top:16px;">The final merged dataset keeps all retained complex rows and all correct rewrites for this question.</p>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    if args.output_path is None:
        args.output_path = dataset_dir / "question_flow.html"

    questions_by_id = load_benchmark_question_map(dataset_dir / "questions.jsonl")
    decisions_by_id = {
        str(payload["question_id"]): payload
        for payload in iter_jsonl(dataset_dir / "question_decisions.jsonl") or []
        if isinstance(payload.get("question_id"), str)
    }

    benchmark_rows_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in iter_jsonl(dataset_dir / "cache" / "benchmark_runs.jsonl") or []:
        question_id = payload.get("question_id")
        if isinstance(question_id, str):
            benchmark_rows_by_qid[question_id].append(payload)

    gepa_rows_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in iter_jsonl(dataset_dir / "gepa_results.jsonl") or []:
        question_deltas = payload.get("question_deltas")
        if not isinstance(question_deltas, list):
            continue
        for delta in question_deltas:
            if not isinstance(delta, dict):
                continue
            question_id = delta.get("question_id")
            if not isinstance(question_id, str):
                continue
            gepa_rows_by_qid[question_id].append(
                {
                    "target_model": payload.get("target_model"),
                    "domain": payload.get("domain"),
                    "run_dir": payload.get("run_dir"),
                    "question_delta": delta,
                }
            )

    def rows_for_question(path: Path) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for payload in iter_jsonl(path) or []:
            meta = payload.get("meta")
            slot_id = meta.get("slot_id") if isinstance(meta, dict) else None
            if not isinstance(slot_id, str):
                continue
            grouped[parse_slot_id(slot_id).question_id].append(payload)
        return grouped

    complex_rows_by_qid = rows_for_question(dataset_dir / "complex_distill.jsonl")
    rewrite_rows_by_qid = rows_for_question(dataset_dir / "rewrite_distill.jsonl")
    final_rows_by_qid = rows_for_question(dataset_dir / "distill_sft.jsonl")

    final_counts = Counter({question_id: len(rows) for question_id, rows in final_rows_by_qid.items()})
    gepa_counts = Counter({question_id: len(rows) for question_id, rows in gepa_rows_by_qid.items()})

    question_id = args.question_id or select_question_id(
        questions_by_id=questions_by_id,
        decisions_by_id=decisions_by_id,
        gepa_counts=gepa_counts,
        final_counts=final_counts,
    )
    if question_id not in questions_by_id:
        raise ValueError(f"Unknown question_id: {question_id}")

    question = questions_by_id[question_id]
    decision = decisions_by_id.get(question_id, {})
    benchmark_rows = sorted(
        benchmark_rows_by_qid.get(question_id, []),
        key=lambda row: (str(row.get("model_name")), int(row.get("sample_index", 0))),
    )
    gepa_rows = gepa_rows_by_qid.get(question_id, [])
    complex_rows = sorted(
        complex_rows_by_qid.get(question_id, []),
        key=lambda row: str(row["meta"]["trajectory_id"]),
    )
    rewrite_rows = sorted(
        rewrite_rows_by_qid.get(question_id, []),
        key=lambda row: str(row["meta"]["trajectory_id"]),
    )
    final_rows = final_rows_by_qid.get(question_id, [])
    summary = load_json(dataset_dir / "pipeline_summary.json")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        build_html(
            dataset_dir=dataset_dir,
            summary=summary,
            question=question,
            decision=decision,
            benchmark_rows=benchmark_rows,
            gepa_rows=gepa_rows,
            complex_rows=complex_rows,
            rewrite_rows=rewrite_rows,
            final_rows=final_rows,
        ),
        encoding="utf-8",
    )
    print(f"question_id={question_id}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
