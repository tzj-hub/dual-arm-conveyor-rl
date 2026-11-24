#!/usr/bin/env python3
"""Plot recorded formal training; does not run physics, training or evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "conveyor-training-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_updates(path: Path) -> list[dict]:
    try:
        lines = path.read_text(errors="replace").splitlines(keepends=True)
    except OSError:
        return []
    records = []
    for line in lines:
        if not line.endswith("\n"):
            continue  # A concurrent partial write is not a completed record.
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validation_rate(record: dict, cohorts: int, objects: int):
    summary = record.get("summary") or {}
    total = cohorts * objects
    sorted_count = summary.get("sorted")
    if (total <= 0 or summary.get("complete") is not True
            or summary.get("requested") != total or len(record.get("episodes", [])) != cohorts
            or not number(sorted_count) or not 0 <= sorted_count <= total):
        return None
    return 100 * sorted_count / total


def pending(ax, text="Pending: no recorded data") -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#666666")


def plot(root: Path, run: Path | None = None) -> tuple[Path, Path]:
    output = run.resolve() if run is not None else root / "results/sequence_set_oracle_cuda_formal"
    if not (output / 'training.json').is_file():
        raise ValueError(f"No recorded training at {output}")
    metadata = read_json(output / "training.json")
    args = metadata.get("arguments", {})
    cohorts, objects = args.get("validation_episodes", 4), args.get("total_objects", 100)
    updates = read_updates(output / "updates.jsonl")
    selection = read_json(output / "selection.json")
    bc = read_json(output / "bc_validation.json").get("evaluation", {})
    # Earlier references use different validation seeds from a new manual run.
    teacher = read_json(root / "results/sequence_teacher_oracle_validation/SUMMARY.json") if run is None else {}
    previous = read_json(root / "results/old_oracle_validation_920000/SUMMARY.json") if run is None else {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle("Sequence / set-context PPO - formal oracle training", fontsize=15)
    ax = axes[0, 0]
    values = []
    for candidate in selection.get("candidates", []):
        rate = validation_rate(candidate, cohorts, objects)
        if rate is not None and number(candidate.get("steps")):
            values.append((candidate["steps"], rate))
    if values:
        ax.plot(*zip(*values), marker="o", color="#2463aa", label="PPO validation")
    references = [("BC validation (supervised)", validation_rate(bc, cohorts, objects), "#cc7d20")]
    if run is None:
        references.append(("Sequence teacher validation",
                           validation_rate(teacher, cohorts, objects) if teacher.get("status") == "complete" else None,
                           "#3b8c63"))
        references.append(("Previous PPO validation",
                           validation_rate(previous, cohorts, objects) if previous.get("status") == "complete" else None,
                           "#775ca6"))
    for label, rate, color in references:
        if rate is not None:
            ax.axhline(rate, color=color, linestyle="--", label=f"{label}: {rate:.2f}%")
    has_validation = bool(values) or any(rate is not None for _, rate, _ in references)
    if has_validation:
        ax.legend(fontsize=8, loc="best")
    else:
        pending(ax, "Pending: complete validation cohorts")
    missing = [name for name, (_, rate, _) in zip(("BC", "teacher", "previous PPO"), references) if rate is None]
    if not values:
        missing.insert(0, "PPO")
    ax.set_title(f"Complete validation ({cohorts} x {objects} objects)" + (f"\nNo complete record: {', '.join(missing)}" if missing else ""))
    ax.set_ylabel("Sorting success (%)")
    ax.set_ylim(0, 100)
    panels = [(axes[0, 1], "ppo_actor_optimizer_steps_total", "Actual cumulative PPO actor updates", "Optimizer steps"),
              (axes[1, 0], "entropy", "Policy entropy (rollout update)", "Entropy (nats)"),
              (axes[1, 1], "explained_variance", "Critic explained variance", "Explained variance")]
    for ax, key, title, ylabel in panels:
        data = [(row["steps"], row[key]) for row in updates
                if number(row.get("steps")) and number(row.get(key))]
        if data:
            ax.plot(*zip(*data), marker=".", linewidth=1.4, color="#2463aa")
        else:
            pending(ax)
        ax.set(title=title, ylabel=ylabel)
        if key == "ppo_actor_optimizer_steps_total":
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    last_step = max([row["steps"] for row in updates if number(row.get("steps"))]
                    + [step for step, _ in values] + [1])
    for ax in axes.flat:
        ax.set_xlabel("PPO environment decision steps")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.set_xlim(0, last_step * 1.03)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.supxlabel(f"Validation is not independent testing. BC updates are excluded from PPO counts.\nSnapshot: {stamp}", fontsize=9)
    output.mkdir(parents=True, exist_ok=True)
    paths = output / "training_curve.png", output / "training_curve.svg"
    for path in paths:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, help="Plot a new manual run without mixing older validation references")
    options = parser.parse_args()
    for destination in plot(options.root.resolve(), options.run):
        print(destination)
