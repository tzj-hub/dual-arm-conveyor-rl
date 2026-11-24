#!/usr/bin/env python3
"""Read the current formal run without importing the simulator or ML stack."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_records(path: Path) -> list[dict]:
    """A concurrently written final line is not a committed record yet."""
    try:
        lines = path.read_text(errors="replace").splitlines(keepends=True)
    except OSError:
        return []
    records = []
    for line in lines:
        if not line.endswith("\n"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def count(record: dict, name: str) -> int:
    value = record.get(name, 0)
    return value if isinstance(value, int) and value >= 0 else 0


def validation_line(label: str, summary: dict) -> str | None:
    if not summary.get("complete") or count(summary, "requested") == 0:
        return None
    sorted_count, total = count(summary, "sorted"), count(summary, "requested")
    return (f"{label}：{sorted_count}/{total} ({100 * sorted_count / total:.2f}%)；"
            f"掉落 {count(summary, 'dropped')}，未完成 {count(summary, 'unfinished')}，"
            f"机器人/臂间异常接触步 {count(summary, 'robot_collision_steps')}/"
            f"{count(summary, 'inter_arm_collision_steps')}")


def report(root: Path, run: Path | None = None) -> str:
    queue = root / "results/formal_training_priority"
    output = run.resolve() if run is not None else root / "results/sequence_set_oracle_cuda_formal"
    log_path = output / "console.log" if run is not None else queue / "training.log"
    state = read_json(output / "launch.json" if run is not None else queue / "STATE.json")
    metadata = read_json(output / "training.json")
    selection = read_json(output / "selection.json")
    bc = read_json(output / "bc_validation.json")
    progress = [r for r in read_records(log_path)
                if r.get("event") == "training_progress"]
    updates = read_records(output / "updates.jsonl")
    latest = progress[-1] if progress else {}
    latest_update = updates[-1] if updates else {}
    args = metadata.get("arguments", {})
    total = count(metadata.get("ppo_config", {}), "steps") or count(args, "steps")
    warmstart_total = count(args, "warmstart_steps")
    warmstarts = [r for r in progress if r.get("phase") == "warmstart"]
    warmstart = warmstarts[-1] if warmstarts else {}
    examples_seen = count(warmstart, "examples_seen") if "examples_seen" in warmstart else count(warmstart, "examples")
    # Evaluation steps restart at zero for each cohort; BC steps are separate.
    ppo_steps = max([count(metadata, "steps"), count(latest_update, "steps")]
                    + [count(r, "steps") for r in progress
                       if r.get("phase") in ("ppo", "ppo_update")])
    actor_updates = max(count(metadata, "actor_optimizer_steps"),
                        count(latest_update, "ppo_actor_optimizer_steps_total"))
    optimizer_updates = max(count(metadata, "ppo_optimizer_steps"),
                            count(latest_update, "ppo_optimizer_steps_total"))
    phase = latest.get("phase")
    queue_status, training_status = state.get("status"), metadata.get("status")
    if queue_status in ("failed", "interrupted") or training_status in ("failed", "interrupted"):
        stage = f"已停止：队列 {queue_status or '未知'} / 训练 {training_status or '未知'}"
    elif queue_status == "teacher":
        stage = "PPO 训练已完成；正在运行同验证集的规则教师对照"
    elif training_status == "complete":
        stage = "训练已完成；以完整验证结果判断质量"
    elif phase == "warmstart":
        stage = ("BC 参数优化中（训练器没有逐 epoch 日志）"
                 if warmstart_total and count(warmstart, "steps") >= warmstart_total
                 else "BC 示范采样中")
    elif phase == "evaluation":
        stage = "PPO 验证中" if actor_updates else "BC 验证中"
        stage += (f"；已完成批次 {count(latest, 'episodes')}，当前种子 {latest.get('seed', '?')}，"
                  f"当前批次决策步 {count(latest, 'steps')}（不计入 PPO 步数）")
    elif phase in ("ppo", "ppo_update"):
        stage = "PPO 采样/更新中"
    else:
        stage = "等待首条进度记录" if not metadata else "初始化/等待下一条进度记录"
    lines = [f"训练进度  {datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"阶段：{stage}",
             f"记录状态：队列 {queue_status or '尚无记录'} / 训练 {training_status or '尚无记录'}",
             f"BC 示范采样：{count(warmstart, 'steps')}/{warmstart_total or '?'} 步，"
             f"累计有效示范 {examples_seen} 条，保留 {count(warmstart, 'examples')} 条",
             f"PPO 采样：{ppo_steps}/{total or '?'} 步；已落盘的实际 actor 更新 {actor_updates} 次，"
             f"PPO optimizer 更新 {optimizer_updates} 次"]
    if latest:
        age = max(0, time.time() - log_path.stat().st_mtime)
        lines.append(f"最近进度事件：{phase}；日志最后写入距今 {age:.0f} 秒（记录状态不等于进程存活检查）")
    validations = []
    bc_line = validation_line("完整 BC 验证", bc.get("evaluation", {}).get("summary", {}))
    if bc_line:
        validations.append(bc_line)
    complete = [candidate for candidate in selection.get("candidates", [])
                if isinstance(candidate, dict) and candidate.get("summary", {}).get("complete")]
    if complete:
        candidate = complete[-1]
        line = validation_line(f"最近完整 PPO 验证（{candidate.get('steps', '?')} 步）", candidate["summary"])
        if line:
            validations.append(line)
    selected = selection.get("selected") or {}
    selected_line = validation_line(f"选中的 PPO（{selected.get('steps', '?')} 步）", selected.get("summary", {}))
    if selected_line:
        validations.append(selected_line)
    lines.extend(validations or ["完整验证成绩：尚未产生；训练 episode 不作为验证成功率。"])
    lines.extend(["以上为验证集成绩，不是独立测试；best.pt 本身不保证超过 BC 或教师。",
                  f"日志：{log_path}", f"更新与模型：{output}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path, help="新手动训练目录；省略则查看之前的正式运行")
    parser.add_argument("--watch", action="store_true", help="每隔指定秒数重新读取；Ctrl-C 仅退出查看")
    parser.add_argument("--interval", type=float, default=15, help="watch 刷新间隔，默认 15 秒")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least one second")
    try:
        while True:
            print(report(args.root.resolve(), args.run), flush=True)
            if not args.watch:
                break
            time.sleep(args.interval)
            print()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
