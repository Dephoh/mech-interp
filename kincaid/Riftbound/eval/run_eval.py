"""Riftbound evaluation runner — composite reward signal for autonomous agents.

Runs all verification checks and produces a machine-readable score:
  - Tests (50%):     pytest pass rate
  - Contracts (30%): serialized state validates against the JSON Schema
  - IR Coverage (20%): percentage of card abilities with effect_ir trees

Usage:
    python eval/run_eval.py           # from the Riftbound project root
    python eval/run_eval.py --json    # output only the JSON result
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
CARD_DEFS_PATH = BACKEND_DIR / "data" / "card_definitions.json"
RESULT_PATH = PROJECT_ROOT / "eval" / "last_result.json"


# ---- pytest runner ----------------------------------------------------


def run_pytest() -> dict:
    """Run the full pytest suite and return pass/fail counts."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-x"],
        capture_output=True,
        text=True,
        cwd=str(BACKEND_DIR),
        timeout=120,
    )
    stdout = result.stdout + result.stderr

    # Parse pytest summary line: "342 passed in 0.99s" or "340 passed, 2 failed in 1.2s"
    passed = 0
    failed = 0
    errors = 0

    m = re.search(r"(\d+)\s+passed", stdout)
    if m:
        passed = int(m.group(1))

    m = re.search(r"(\d+)\s+failed", stdout)
    if m:
        failed = int(m.group(1))

    m = re.search(r"(\d+)\s+error", stdout)
    if m:
        errors = int(m.group(1))

    total = passed + failed + errors
    score = (passed / total * 100) if total > 0 else 0.0

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "score": round(score, 1),
        "exit_code": result.returncode,
    }


# ---- Contract check ---------------------------------------------------


def run_contracts() -> dict:
    """Run only the contract tests and return pass/fail."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_contracts.py", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(BACKEND_DIR),
        timeout=60,
    )
    stdout = result.stdout + result.stderr
    passed = 0
    failed = 0

    m = re.search(r"(\d+)\s+passed", stdout)
    if m:
        passed = int(m.group(1))

    m = re.search(r"(\d+)\s+failed", stdout)
    if m:
        failed = int(m.group(1))

    valid = result.returncode == 0 and failed == 0
    return {
        "valid": valid,
        "passed": passed,
        "failed": failed,
        "score": 100.0 if valid else 0.0,
        "detail": stdout.strip().split("\n")[-1] if stdout.strip() else "",
    }


# ---- IR coverage ------------------------------------------------------


def measure_ir_coverage() -> dict:
    """Count abilities with effect_ir vs text-only in card_definitions.json."""
    if not CARD_DEFS_PATH.exists():
        return {"with_ir": 0, "text_only": 0, "total": 0, "pct": 0.0, "score": 0.0}

    with open(CARD_DEFS_PATH, encoding="utf-8") as f:
        cards = json.load(f)

    # card_definitions.json is a list of card dicts
    if isinstance(cards, dict):
        card_list = cards.values()
    else:
        card_list = cards

    with_ir = 0
    text_only = 0

    for card in card_list:
        for ab in card.get("abilities", []):
            if ab.get("effect_ir"):
                with_ir += 1
            elif ab.get("text", "").strip():
                text_only += 1
            # Abilities with neither IR nor text are considered structural (no coverage issue)

    total = with_ir + text_only
    pct = (with_ir / total * 100) if total > 0 else 100.0

    return {
        "with_ir": with_ir,
        "text_only": text_only,
        "total": total,
        "pct": round(pct, 1),
        "score": round(pct, 1),
    }


# ---- Composite score --------------------------------------------------


WEIGHTS = {
    "tests": 0.50,
    "contracts": 0.30,
    "ir_coverage": 0.20,
}


def compute_composite(tests: dict, contracts: dict, ir_coverage: dict) -> float:
    return round(
        WEIGHTS["tests"] * tests["score"]
        + WEIGHTS["contracts"] * contracts["score"]
        + WEIGHTS["ir_coverage"] * ir_coverage["score"],
        1,
    )


# ---- Main -------------------------------------------------------------


def main():
    json_only = "--json" in sys.argv

    if not json_only:
        print("=" * 60)
        print(" RIFTBOUND EVAL")
        print("=" * 60)
        print()

    # 1. Run all pytest tests
    if not json_only:
        print("Running tests ...", flush=True)
    tests = run_pytest()

    # 2. Run contract validation (subset of tests, but reported separately)
    if not json_only:
        print("Checking contracts ...", flush=True)
    contracts = run_contracts()

    # 3. Measure IR coverage
    if not json_only:
        print("Measuring IR coverage ...", flush=True)
    ir_cov = measure_ir_coverage()

    # 4. Composite
    composite = compute_composite(tests, contracts, ir_cov)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": tests,
        "contracts": contracts,
        "ir_coverage": ir_cov,
        "composite_score": composite,
    }

    # Save result
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if json_only:
        print(json.dumps(result, indent=2))
    else:
        print()
        print(f"{'Section':<20} {'Result':<30} {'Score':>8}")
        print("-" * 60)
        print(f"{'Tests':<20} {tests['passed']}/{tests['total']} passed{'':<15} {tests['score']:>7.1f}")
        print(f"{'Contracts':<20} {'PASS' if contracts['valid'] else 'FAIL':<30} {contracts['score']:>7.1f}")
        print(f"{'IR Coverage':<20} {ir_cov['with_ir']}/{ir_cov['total']} abilities{'':<12} {ir_cov['score']:>7.1f}")
        print("-" * 60)
        print(f"{'COMPOSITE':<20} {'':<30} {composite:>7.1f}")
        print("=" * 60)
        print(f"\nResult saved to {RESULT_PATH.relative_to(PROJECT_ROOT)}")

    # Exit with non-zero if tests failed
    sys.exit(0 if tests["exit_code"] == 0 else 1)


if __name__ == "__main__":
    main()
