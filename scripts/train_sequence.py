#!/usr/bin/env python3
"""Preview a manual training run; only --execute starts the trainer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(options):
    config = json.loads(options.config.read_text())
    arguments = dict(config["training_arguments"])
    output = (options.output or ROOT / "results" / (
        "manual_sequence_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    arguments["output"] = str(output)
    for key in ("steps", "device", "warmstart_steps", "warmstart_max_examples"):
        value = getattr(options, key)
        if value is not None:
            arguments[key] = value
    if options.steps is not None:
        arguments["expert_decay_steps"] = options.steps
    if options.expert_balance is not None:
        arguments["expert_balance"] = options.expert_balance
    if options.initialize_from:
        checkpoint = options.initialize_from.resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Initialization checkpoint missing: {checkpoint}")
        arguments.update(initialize_from=str(checkpoint), warmstart_steps=0,
                         warmstart_max_examples=None, validate_bc=False)
    if arguments.get("resume") is not None:
        raise ValueError("This launcher creates a new run; use resume_scheduler.sh for a compatible interrupted run")
    if arguments["steps"] <= 0 or arguments["warmstart_steps"] < 0:
        raise ValueError("PPO steps must be positive and BC steps nonnegative")
    cap = arguments.get("warmstart_max_examples")
    if cap is not None and cap <= 0:
        raise ValueError("BC retained-example limit must be positive")
    argv = []
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            argv.extend([flag, *map(str, value)])
        else:
            argv.append(f"{flag}={value}")
    command = ["bash", "-c", 'set -e; source "$1"; shift; exec "$@"', "sequence-manual",
               str(ROOT / "scripts/runtime_env.sh"), sys.executable, "-u", "-m",
               "dual_arm_conveyor.hierarchical_train", *argv]
    return dict(status="preview", output=str(output), arguments=arguments, command=command,
                command_preview=shlex.join(command),
                initialization="weights only; fresh optimizer/counters" if options.initialize_from else "fresh BC then PPO",
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / "src/dual_arm_conveyor").glob("*.py")},
                config_sha256=hashlib.sha256(options.config.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sequence_r3_fresh.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--warmstart-steps", type=int)
    parser.add_argument("--warmstart-max-examples", type=int)
    parser.add_argument("--expert-balance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually start this run; default is preview only")
    options = parser.parse_args()
    try:
        plan = prepare(options)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not options.execute:
        print("仅预览；未创建输出目录、未导入训练框架、未启动训练。加 --execute 才会运行。")
        return 0
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    plan.update(status="running", started_utc=datetime.now(timezone.utc).isoformat())
    record = output / "launch.json"
    record.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.setdefault("CONVEYOR_LIBM_COMPAT", "1")
    child = None
    try:
        with (output / "console.log").open("x") as log:
            child = subprocess.Popen(plan["command"], cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = child.wait()
        plan.update(status="complete" if code == 0 else "failed", returncode=code)
    except BaseException as error:
        plan.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=repr(error))
        if child is not None and child.poll() is None:
            for signum, grace in ((signal.SIGINT, 30), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                if child.poll() is not None:
                    break
                try:
                    os.killpg(child.pid, signum)
                    child.wait(timeout=grace)
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        plan["ended_utc"] = datetime.now(timezone.utc).isoformat()
        record.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
