#!/usr/bin/env python3
"""Sanity checks for a notebook-only repository — run by CI and by hand.

    python scripts/check_repo.py

This repository ships notebooks, figures and a results table, so there is no app to
boot. What *can* silently rot is checked here:

1. every notebook is well-formed JSON with an nbformat 4 structure and real cells
   (a truncated or badly merged notebook is the classic failure — it opens as an
   error in Jupyter and the work looks lost);
2. every relative link and image in the README exists in the tree (moved figures
   are the usual cause of a README full of broken images);
3. requirements.txt is parseable and every pinned name is a valid package spec;
4. the results CSV parses and its header matches what the README promises.

Exit code 0 means all good; anything else means the check failed.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
failures: list[str] = []
checks = 0


def ok(message: str) -> None:
    global checks
    checks += 1
    print(f"  ok   {message}")


def bad(message: str) -> None:
    failures.append(message)
    print(f"  FAIL {message}")


def check_notebooks() -> None:
    print("\n[1] notebooks are valid nbformat-4 documents")
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        bad("no notebooks found under notebooks/")
        return
    for path in paths:
        rel = path.relative_to(ROOT)
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            bad(f"{rel} is not valid JSON ({exc})")
            continue
        problems = []
        if nb.get("nbformat") != 4:
            problems.append(f"nbformat={nb.get('nbformat')!r}, expected 4")
        cells = nb.get("cells")
        if not isinstance(cells, list) or not cells:
            problems.append("no cells")
            cells = []
        code = 0
        for i, cell in enumerate(cells):
            if cell.get("cell_type") not in {"code", "markdown", "raw"}:
                problems.append(f"cell {i} has cell_type={cell.get('cell_type')!r}")
            source = cell.get("source")
            if not isinstance(source, (str, list)):
                problems.append(f"cell {i} has no usable source")
            if cell.get("cell_type") == "code":
                code += 1
        if code == 0:
            problems.append("no code cells")
        metadata = nb.get("metadata") or {}
        if "kernelspec" not in metadata:
            problems.append("missing kernelspec metadata")
        if problems:
            bad(f"{rel}: " + "; ".join(problems))
        else:
            ok(f"{rel} ({len(cells)} cells, {code} code)")


LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")


def check_readme_links() -> None:
    print("\n[2] every relative README link resolves")
    readme = ROOT / "README.md"
    if not readme.exists():
        bad("README.md is missing")
        return
    text = readme.read_text(encoding="utf-8")
    targets = [t for t in LINK.findall(text) if not re.match(r"^(https?:|mailto:|#)", t)]
    if not targets:
        bad("README links no figures or results at all")
        return
    seen = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        if (ROOT / target.split("#")[0]).exists():
            ok(f"{target} exists")
        else:
            bad(f"README points at {target}, which is not in the repository")


def check_requirements() -> None:
    print("\n[3] requirements.txt is parseable")
    path = ROOT / "requirements.txt"
    if not path.exists():
        bad("requirements.txt is missing")
        return
    spec = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?\s*([<>=!~]=?[^;#]*)?$")
    names: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not spec.match(line):
            bad(f"requirements.txt line {number} is not a valid requirement: {raw!r}")
        names.append(re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0])
    if names:
        ok(f"{len(names)} requirements: {', '.join(names)}")
    else:
        bad("requirements.txt lists no packages")


def check_results() -> None:
    print("\n[4] the results table parses")
    path = ROOT / "results" / "final-model-comparison.csv"
    if not path.exists():
        bad("results/final-model-comparison.csv is missing")
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        bad("results CSV has no data rows")
        return
    header = [h.strip().lower() for h in rows[0]]
    missing = [c for c in ("accuracy", "f1", "auc") if not any(c in h for h in header)]
    if missing:
        bad(f"results CSV header is missing columns: {missing} (header={rows[0]})")
    else:
        ok(f"{len(rows) - 1} result rows, header {rows[0]}")


def main() -> int:
    print(f"checking {ROOT}")
    check_notebooks()
    check_readme_links()
    check_requirements()
    check_results()
    print()
    if failures:
        print(f"FAILED - {len(failures)} problem(s), {checks} check(s) passed")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"OK - {checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
