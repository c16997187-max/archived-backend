"""Measure OCR speed and memory on this machine (Phase 7: "your panel will ask").

    python -m scripts.bench_ocr
    python -m scripts.bench_ocr --forms 20 --handwriting both --json private_data/bench.json

Fills the enrollment template with made-up values in a script font, tilts and blurs it a
little like a phone photo, and runs steps 1 to 9 on each form. Each handwriting model runs
in its own process, so one model's memory never counts against the other. Reports model
load time, seconds per form (mean and 95th percentile, and the preprocessing part), peak
memory of the process, and the character error rate and exact-field rate of the
normalized values against the normalized values written (so case alone is no error in
a title-cased name).

The accuracy numbers come from a computer font, not handwriting: use them to compare the
models' speed and to catch regressions, and the Step 0 test set (scripts/ocr_debug.py
--predictions, then scripts/eval_ocr.py) for accuracy claims.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

from app.config import get_settings
from app.ocr.model_files import ModelsMissing
from app.ocr.models import load_engine, physical_cores
from app.ocr.normalize import normalize_field
from app.ocr.preprocess import TemplateCandidate, preprocess
from app.ocr.recognize import recognize_crops
from app.ocr.templates import FormLayout, load_layout_file
from scripts.eval_ocr import edit_distance
from scripts.make_placeholder_form import render_form

TEMPLATE = Path("docs/forms/enrollment_v1.json")
FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

# Made up. Combined at random, so no form matches a real person.
FIRST = ["Maria", "Jose", "Ana", "Paolo", "Liza", "Mark", "Grace", "Ramon", "Joy", "Carlo"]
LAST = ["Santos", "Reyes", "Bautista", "Garcia", "Mendoza", "Torres", "Ramos", "Flores"]
PLACES = ["Baguio City", "La Trinidad", "Itogon", "Tuba", "Sablan"]
STREETS = ["Purok 3", "Purok 7", "Km 4", "Lower Rock", "Upper Session"]


def _values(rng: np.random.Generator) -> dict[str, str]:
    def pick(items: list[str]) -> str:
        return str(rng.choice(items))

    year = int(rng.integers(2010, 2020))
    return {
        "school_year": "2026-2027",
        "grade_level": f"Grade {int(rng.integers(1, 11))}",
        "lrn": "".join(str(d) for d in rng.integers(0, 10, 12)),
        "last_name": pick(LAST),
        "first_name": pick(FIRST),
        "middle_name": pick(LAST),
        "extension": "",
        "birth_date": f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}/{year}",
        "place_of_birth": pick(PLACES),
        "mother_tongue": "Ilocano",
        "religion": "Adventist",
        "ip_group": "",
        "house_street": pick(STREETS),
        "barangay": pick(STREETS),
        "municipality": pick(PLACES),
        "province": "Benguet",
        "contact_number": "09" + "".join(str(d) for d in rng.integers(0, 10, 9)),
        "father_name": f"{pick(FIRST)} {pick(LAST)}",
        "mother_name": f"{pick(FIRST)} {pick(LAST)}",
        "guardian_name": "",
        "guardian_relationship": "",
    }


def make_form(layout: FormLayout, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, str]]:
    """A filled, slightly tilted and blurred form, and the values written on it."""
    page = cv2.cvtColor(render_form(layout), cv2.COLOR_GRAY2BGR)
    values = _values(rng)
    written = {}
    for name, box in layout.boxes().items():
        if box.kind == "checkbox":
            continue
        text = values.get(name, "")
        written[name] = text
        if text:
            ink = (int(rng.integers(90, 140)), 40, 20)  # blue-black pen
            cv2.putText(page, text, (box.x + 15, box.y + box.h - 22), FONT, 1.3, ink, 2)
    ticked = layout.boxes()[str(rng.choice(["sex_male", "sex_female"]))]
    cv2.line(page, (ticked.x + 8, ticked.y + 20), (ticked.x + 18, ticked.y + 32), (20, 20, 20), 4)
    cv2.line(page, (ticked.x + 18, ticked.y + 32), (ticked.x + 34, ticked.y + 6), (20, 20, 20), 4)
    height, width = page.shape[:2]
    tilt = np.float64([[1, 0.01, 12], [-0.01, 1, -8], [0, 0, 1]])
    page = cv2.warpPerspective(page, tilt, (width, height), borderValue=(235, 235, 235))
    page = cv2.GaussianBlur(page, (3, 3), 0)
    noise = rng.normal(0, 6, page.shape)
    return np.clip(page + noise, 0, 255).astype(np.uint8), written


def peak_memory_mb() -> float:
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KiB
    return peak / 2**20


def run_one(handwriting: str, forms: int, tier: str, threads: int | None) -> dict[str, Any]:
    settings = get_settings()
    layout = load_layout_file(TEMPLATE)
    candidate = TemplateCandidate(None, layout, TEMPLATE.parent / layout.image)
    rng = np.random.default_rng(7)

    started = time.perf_counter()
    engine = load_engine(settings.models_dir, tier, handwriting, threads)
    load_seconds = time.perf_counter() - started

    seconds, prep_seconds, errors, chars, exact, fields = [], [], 0, 0, 0, 0
    with tempfile.TemporaryDirectory() as work:
        for i in range(forms):
            image, written = make_form(layout, rng)
            path = Path(work) / f"form_{i}.png"
            cv2.imwrite(str(path), image)
            started = time.perf_counter()
            prepared = preprocess(path, None, [candidate])
            prep_seconds.append(time.perf_counter() - started)
            results = recognize_crops(prepared.crops, engine, settings.ocr_confidence_threshold)
            seconds.append(time.perf_counter() - started)
            read = {r.name: r.normalized or "" for r in results}
            for name, text in written.items():
                truth = normalize_field(name, text).value or ""
                errors += edit_distance(truth, read.get(name, ""))
                chars += len(truth)
                exact += truth == read.get(name, "")
                fields += 1
    return {
        "handwriting": handwriting,
        "model_versions": engine.versions,
        "tier": tier,
        "threads": engine.threads,
        "forms": forms,
        "load_seconds": round(load_seconds, 2),
        "seconds_per_form_mean": round(statistics.mean(seconds), 2),
        "seconds_per_form_p95": round(float(np.percentile(seconds, 95)), 2),
        "preprocess_seconds_mean": round(statistics.mean(prep_seconds), 2),
        "peak_memory_mb": round(peak_memory_mb()),
        "synthetic_cer": round(errors / max(chars, 1), 4),
        "synthetic_exact_fields": round(exact / max(fields, 1), 4),
    }


def machine() -> dict[str, Any]:
    return {
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": physical_cores(),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
    }


def format_results(info: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        f"Machine: {info['cpu']}, {info['physical_cores']} physical / "
        f"{info['logical_cores']} logical cores, {info['ram_gb']} GB RAM, {info['os']}",
        f"{'handwriting':<12} {'load s':>7} {'s/form':>7} {'p95 s':>7} {'prep s':>7} "
        f"{'peak MB':>8} {'CER*':>7} {'exact*':>7}",
    ]
    for r in results:
        lines.append(
            f"{r['handwriting']:<12} {r['load_seconds']:>7} {r['seconds_per_form_mean']:>7} "
            f"{r['seconds_per_form_p95']:>7} {r['preprocess_seconds_mean']:>7} "
            f"{r['peak_memory_mb']:>8} "
            f"{r['synthetic_cer']:>7.1%} {r['synthetic_exact_fields']:>7.1%}"
        )
    first = results[0] if results else {}
    lines.append(
        f"{first.get('forms', 0)} generated forms each, PP-OCRv6 {first.get('tier', '')}, "
        f"{first.get('threads', '')} ONNX Runtime threads."
    )
    lines.append("* On a computer script font, not handwriting. Use the Step 0 set for accuracy.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the OCR pipeline on this machine.")
    parser.add_argument("--forms", type=int, default=10)
    parser.add_argument("--handwriting", choices=("trocr", "ppocr", "both"), default="both")
    parser.add_argument("--tier", choices=("tiny", "small", "medium"), default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--json", type=Path, help="Also write the results to this file")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    tier = args.tier or get_settings().ocr_det_tier

    if args.child:
        try:
            result = run_one(args.handwriting, args.forms, tier, args.threads)
        except ModelsMissing as exc:
            print(exc, file=sys.stderr)
            return 1
        print(json.dumps(result))
        return 0

    models = ("trocr", "ppocr") if args.handwriting == "both" else (args.handwriting,)
    results = []
    for model in models:
        command = [sys.executable, "-m", "scripts.bench_ocr", "--child", "--handwriting", model,
                   "--forms", str(args.forms), "--tier", tier]  # fmt: skip
        if args.threads:
            command += ["--threads", str(args.threads)]
        print(f"Running {model} on {args.forms} forms...", file=sys.stderr)
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            print(
                done.stderr.strip().splitlines()[-1] if done.stderr else "failed", file=sys.stderr
            )
            return 1
        results.append(json.loads(done.stdout.strip().splitlines()[-1]))
    info = machine()
    print(format_results(info, results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"machine": info, "results": results}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
