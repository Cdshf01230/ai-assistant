#!/usr/bin/env python3
"""Run and summarize the public benchmark checks for the MVP project."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/home/ubuntu/ai-assistant")
DATA_ROOT = PROJECT_ROOT / "benchmarks" / "data"
DEFAULT_REPORT = PROJECT_ROOT / "benchmarks" / "benchmark_report.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "benchmarks" / "benchmark_report.md"


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def has_files(path: Path, pattern: str) -> int:
    return sum(1 for item in path.rglob(pattern) if item.is_file()) if path.exists() else 0


def check_datasets() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    diode = DATA_ROOT / "diode"
    diode_npy = has_files(diode, "*.npy")
    diode_png = has_files(diode, "*.png")
    results["diode"] = {
        "status": "ok" if diode_npy and diode_png else "missing",
        "path": str(diode),
        "size_bytes": size_bytes(diode) if diode.exists() else 0,
        "depth_arrays": diode_npy,
        "rgb_images": diode_png,
    }

    nyu = DATA_ROOT / "nyu" / "nyu_depth_v2_labeled.mat"
    nyu_info: dict[str, Any] = {"path": str(nyu), "size_bytes": size_bytes(nyu) if nyu.exists() else 0}
    if nyu.exists():
        try:
            import h5py

            with h5py.File(nyu, "r") as handle:
                required = {"images", "depths", "labels", "instances"}
                missing = sorted(required - set(handle.keys()))
                nyu_info.update(
                    {
                        "status": "ok" if not missing else "invalid",
                        "keys": sorted(handle.keys()),
                        "missing_keys": missing,
                        "images_shape": list(handle["images"].shape),
                        "depths_shape": list(handle["depths"].shape),
                    }
                )
        except Exception as exc:  # Keep one broken dataset from hiding others.
            nyu_info.update({"status": "invalid", "error": repr(exc)})
    else:
        nyu_info["status"] = "missing"
    results["nyu_depth_v2"] = nyu_info

    sun = DATA_ROOT / "sunrgbd" / "extracted"
    sun_files = has_files(sun, "*")
    results["sun_rgbd"] = {
        "status": "ok" if sun_files else "missing",
        "path": str(sun),
        "size_bytes": size_bytes(sun) if sun.exists() else 0,
        "files": sun_files,
    }

    coco = DATA_ROOT / "coco"
    coco_json = sorted(str(item.relative_to(coco)) for item in coco.rglob("*.json")) if coco.exists() else []
    results["coco"] = {
        "status": "ok" if len(coco_json) >= 6 else "missing",
        "path": str(coco),
        "size_bytes": size_bytes(coco) if coco.exists() else 0,
        "annotation_files": coco_json,
    }

    fleurs = DATA_ROOT / "fleurs" / "hf"
    fleurs_files = has_files(fleurs, "*")
    results["fleurs_vi"] = {
        "status": "ok" if fleurs_files else "missing",
        "path": str(fleurs),
        "size_bytes": size_bytes(fleurs) if fleurs.exists() else 0,
        "cached_files": fleurs_files,
        "validation_rows": 361,
    }

    try:
        import ai2thor  # noqa: F401

        ai2thor_status = "ok"
    except Exception as exc:
        ai2thor_status = f"invalid: {exc!r}"
    results["ai2thor"] = {"status": ai2thor_status, "package": "ai2thor"}
    return results


def run_command(name: str, command: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
    return {
        "name": name,
        "command": command,
        "exit_code": process.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def run_project_benchmarks() -> list[dict[str, Any]]:
    python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    return [
        run_command("vlm_eval_4b", [python, "scripts/09_eval_vlm.py"]),
        run_command(
            "depth_probe_outdoor",
            [python, "scripts/11_depth_probe.py", "--dump", "benchmarks/depth_outdoor.json"],
        ),
        run_command(
            "depth_probe_indoor",
            [python, "scripts/11_depth_probe.py", "--which", "indoor", "--dump", "benchmarks/depth_indoor.json"],
        ),
    ]


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Started: `{report['started_utc']}`",
        "",
        "## Dataset checks",
        "",
        "| Dataset | Status | Details |",
        "|---|---|---|",
    ]
    for name, result in report["datasets"].items():
        details = ", ".join(f"{key}={value}" for key, value in result.items() if key not in {"status", "path"})
        lines.append(f"| `{name}` | `{result['status']}` | {details} |")
    if report["commands"]:
        lines.extend(["", "## Project benchmark commands", ""])
        for command in report["commands"]:
            lines.extend(
                [
                    f"### {command['name']}",
                    "",
                    f"- Exit code: `{command['exit_code']}`",
                    f"- Duration: `{command['duration_seconds']} s`",
                    "",
                    "```text",
                    command["stdout"].rstrip(),
                    command["stderr"].rstrip(),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("inventory", "project", "all"), default="inventory")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "mode": args.mode,
        "started_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "datasets": check_datasets(),
        "commands": run_project_benchmarks() if args.mode in {"project", "all"} else [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"JSON report -> {args.report}")
    print(f"Markdown report -> {args.markdown}")
    return 0 if all(item["status"] == "ok" for item in report["datasets"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())