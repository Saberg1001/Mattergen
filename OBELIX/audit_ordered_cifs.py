#!/usr/bin/env python3
"""Audit ordered CIF candidates and merge results into the CIF inventory."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifParser


AUDIT_FIELDS = (
    "ordering_batch_status",
    "ordering_result_summary",
    "ordered_candidate_count",
    "geometry_valid_candidate_count",
    "geometry_error_candidate_count",
    "selected_ordered_file",
    "best_error_file",
    "selected_ordered_formula",
    "selected_ordered_num_atoms",
    "ordering_supercell_multiplier",
    "ordering_composition_max_drift",
    "ordering_split_pair_count",
    "ordering_split_cluster_count",
    "geometry_min_distance_angstrom",
    "geometry_min_distance_pair",
    "geometry_min_radius_ratio",
    "geometry_min_radius_pair",
    "geometry_min_li_li_distance_angstrom",
    "geometry_min_o_o_distance_angstrom",
    "geometry_min_o_f_distance_angstrom",
    "geometry_issues",
)


@dataclass(frozen=True)
class CandidateAudit:
    material_id: str
    source_path: Path
    structure: Structure | None
    formula: str
    num_atoms: int | None
    minimum_distance: float | None
    minimum_distance_pair: str
    minimum_radius_ratio: float | None
    minimum_radius_pair: str
    minimum_li_li_distance: float | None
    minimum_o_o_distance: float | None
    minimum_o_f_distance: float | None
    issues: tuple[str, ...]
    report_row: dict[str, str]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ordering-report", type=Path, nargs="+", required=True)
    parser.add_argument("--promote-dir", type=Path, required=True)
    parser.add_argument("--error-dir", type=Path, default=None)
    parser.add_argument("--min-distance", type=float, default=0.8)
    parser.add_argument("--min-radius-ratio", type=float, default=0.75)
    parser.add_argument("--min-li-li-distance", type=float, default=1.8)
    parser.add_argument("--min-o-o-distance", type=float, default=1.2)
    parser.add_argument("--min-o-f-distance", type=float, default=1.2)
    return parser


def covalent_radius(symbol: str) -> float:
    radius = CovalentRadius.radius.get(symbol)
    if radius is not None:
        return float(radius)
    atomic_radius = Element(symbol).atomic_radius
    return float(atomic_radius) if atomic_radius is not None else 1.0


def minimum_pair_metric(
    structure: Structure,
    values: np.ndarray,
) -> tuple[float, str]:
    first, second = np.unravel_index(np.argmin(values), values.shape)
    pair = "-".join(
        sorted((structure[first].specie.symbol, structure[second].specie.symbol))
    )
    return float(values[first, second]), pair


def species_pair_minimum_distance(
    structure: Structure,
    distances: np.ndarray,
    first_symbol: str,
    second_symbol: str,
) -> float | None:
    """Smallest distance between two element subsets, independent of the global minimum.

    ``distances`` is expected to have its diagonal filled with ``inf`` so that a
    same-element query never returns a site's zero self-distance.
    """
    first = [
        index
        for index, site in enumerate(structure)
        if site.specie.symbol == first_symbol
    ]
    if first_symbol == second_symbol:
        if len(first) < 2:
            return None
        return float(np.min(distances[np.ix_(first, first)]))
    second = [
        index
        for index, site in enumerate(structure)
        if site.specie.symbol == second_symbol
    ]
    if not first or not second:
        return None
    return float(np.min(distances[np.ix_(first, second)]))


def audit_candidate(row: dict[str, str], args: argparse.Namespace) -> CandidateAudit:
    material_id = row["parent_id"]
    path = Path(row["output_file"])
    issues: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structures = CifParser(path).parse_structures(primitive=False)
        if not structures:
            raise ValueError("No structure parsed from CIF")
        structure = structures[0]
    except Exception as exc:
        return CandidateAudit(
            material_id=material_id,
            source_path=path,
            structure=None,
            formula="",
            num_atoms=None,
            minimum_distance=None,
            minimum_distance_pair="",
            minimum_radius_ratio=None,
            minimum_radius_pair="",
            minimum_li_li_distance=None,
            minimum_o_o_distance=None,
            minimum_o_f_distance=None,
            issues=(f"parse_error:{type(exc).__name__}",),
            report_row=row,
        )

    if not structure.is_ordered:
        issues.append("not_ordered")
    if row.get("within_composition_tolerance") != "True":
        issues.append("composition_drift")

    distances = np.asarray(structure.distance_matrix, dtype=float)
    np.fill_diagonal(distances, np.inf)
    minimum_distance, minimum_distance_pair = minimum_pair_metric(
        structure, distances
    )

    radii = np.asarray(
        [covalent_radius(site.specie.symbol) for site in structure], dtype=float
    )
    radius_ratios = distances / (radii[:, None] + radii[None, :])
    minimum_radius_ratio, minimum_radius_pair = minimum_pair_metric(
        structure, radius_ratios
    )

    minimum_li_li_distance = species_pair_minimum_distance(
        structure, distances, "Li", "Li"
    )
    minimum_o_o_distance = species_pair_minimum_distance(
        structure, distances, "O", "O"
    )
    minimum_o_f_distance = species_pair_minimum_distance(
        structure, distances, "O", "F"
    )

    if minimum_distance < args.min_distance:
        issues.append(f"distance<{args.min_distance:g}")
    if minimum_radius_ratio < args.min_radius_ratio:
        issues.append(f"radius_ratio<{args.min_radius_ratio:g}")
    if (
        minimum_li_li_distance is not None
        and minimum_li_li_distance < args.min_li_li_distance
    ):
        issues.append(f"Li-Li<{args.min_li_li_distance:g}")
    if (
        minimum_o_o_distance is not None
        and minimum_o_o_distance < args.min_o_o_distance
    ):
        issues.append(f"O-O<{args.min_o_o_distance:g}")
    if (
        minimum_o_f_distance is not None
        and minimum_o_f_distance < args.min_o_f_distance
    ):
        issues.append(f"O-F<{args.min_o_f_distance:g}")

    return CandidateAudit(
        material_id=material_id,
        source_path=path,
        structure=structure,
        formula=structure.composition.formula,
        num_atoms=len(structure),
        minimum_distance=minimum_distance,
        minimum_distance_pair=minimum_distance_pair,
        minimum_radius_ratio=minimum_radius_ratio,
        minimum_radius_pair=minimum_radius_pair,
        minimum_li_li_distance=minimum_li_li_distance,
        minimum_o_o_distance=minimum_o_o_distance,
        minimum_o_f_distance=minimum_o_f_distance,
        issues=tuple(issues),
        report_row=row,
    )


def candidate_sort_key(candidate: CandidateAudit) -> tuple[bool, int, float, float, float]:
    return (
        candidate.is_valid,
        -len(candidate.issues),
        candidate.minimum_radius_ratio or -math.inf,
        candidate.minimum_li_li_distance or math.inf,
        candidate.minimum_distance or -math.inf,
    )


def output_path(output_dir: Path, material_id: str, index: int) -> Path:
    suffix = "" if index == 0 else f"_{index:02d}"
    return output_dir / f"{material_id}_ordered{suffix}.cif"


def promote_candidates(
    candidates: Iterable[CandidateAudit],
    output_dir: Path,
) -> list[Path]:
    valid = sorted(
        (candidate for candidate in candidates if candidate.is_valid),
        key=candidate_sort_key,
        reverse=True,
    )
    paths = [
        output_path(output_dir, candidate.material_id, index)
        for index, candidate in enumerate(valid)
    ]
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".audit-copy-") as name:
        temporary_dir = Path(name)
        staged_paths: list[Path] = []
        for index, candidate in enumerate(valid):
            staged = temporary_dir / f"{index:04d}.cif"
            shutil.copy2(candidate.source_path, staged)
            staged_paths.append(staged)
        for existing in output_dir.glob(f"{candidates[0].material_id}_ordered*.cif"):
            existing.unlink()
        for staged, destination in zip(staged_paths, paths):
            shutil.copy2(staged, destination)
    return paths


def promote_error_candidates(
    candidates: Iterable[CandidateAudit],
    output_dir: Path | None,
) -> list[Path]:
    if output_dir is None:
        return []
    invalid = sorted(
        (candidate for candidate in candidates if not candidate.is_valid),
        key=candidate_sort_key,
        reverse=True,
    )
    copyable = [candidate for candidate in invalid if candidate.source_path.is_file()]
    paths = [
        output_path(output_dir, candidate.material_id, index)
        for index, candidate in enumerate(copyable)
    ]
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".audit-copy-") as name:
        temporary_dir = Path(name)
        staged_paths: list[Path] = []
        for index, candidate in enumerate(copyable):
            staged = temporary_dir / f"{index:04d}.cif"
            shutil.copy2(candidate.source_path, staged)
            staged_paths.append(staged)
        material_id = candidates[0].material_id
        for existing in output_dir.glob(f"{material_id}_ordered*.cif"):
            existing.unlink()
        for staged, destination in zip(staged_paths, paths):
            shutil.copy2(staged, destination)
    return paths


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.8f}"


def audit_values(
    candidates: list[CandidateAudit],
    promoted_paths: list[Path],
    error_paths: list[Path],
) -> dict[str, str]:
    selected = max(candidates, key=candidate_sort_key)
    valid_count = sum(candidate.is_valid for candidate in candidates)
    issue_text = ";".join(selected.issues)
    report_statuses = {candidate.report_row.get("status", "") for candidate in candidates}
    source_was_ordered = report_statuses == {"copied_ordered"}
    generation_failed = bool(report_statuses) and report_statuses <= {
        "error",
        "ranking_error",
    }
    if valid_count:
        if source_was_ordered:
            result_summary = "原始 CIF 已完全有序，并通过几何检查"
        else:
            result_summary = (
                f"成功有序化（{valid_count}/{len(candidates)} "
                "个候选通过几何检查）"
            )
    elif generation_failed:
        messages = sorted(
            {
                candidate.report_row.get("message", "")
                for candidate in candidates
                if candidate.report_row.get("message", "")
            }
        )
        reason = "; ".join(messages) or "未生成可审计的 CIF"
        result_summary = f"有序化生成失败，需人工处理（原因：{reason}）"
    else:
        reason = issue_text or "几何检查未通过"
        prefix = "原始 CIF 已有序，但" if source_was_ordered else ""
        result_summary = (
            f"{prefix}未通过几何检查（0/{len(candidates)} 个候选通过；"
            f"原因：{reason}）"
        )
    row = selected.report_row
    return {
        "ordering_batch_status": (
            "source_ordered"
            if valid_count and source_was_ordered
            else "geometry_ok"
            if valid_count
            else "ordering_error"
            if generation_failed
            else "manual_review"
        ),
        "ordering_result_summary": result_summary,
        "ordered_candidate_count": str(len(candidates)),
        "geometry_valid_candidate_count": str(valid_count),
        "geometry_error_candidate_count": str(len(candidates) - valid_count),
        "selected_ordered_file": str(promoted_paths[0]) if promoted_paths else "",
        "best_error_file": str(error_paths[0]) if error_paths else "",
        "selected_ordered_formula": selected.formula,
        "selected_ordered_num_atoms": (
            "" if selected.num_atoms is None else str(selected.num_atoms)
        ),
        "ordering_supercell_multiplier": row.get("supercell_multiplier", ""),
        "ordering_composition_max_drift": row.get("composition_max_drift", ""),
        "ordering_split_pair_count": row.get("split_site_pair_count", ""),
        "ordering_split_cluster_count": row.get("split_site_cluster_count", ""),
        "geometry_min_distance_angstrom": format_float(selected.minimum_distance),
        "geometry_min_distance_pair": selected.minimum_distance_pair,
        "geometry_min_radius_ratio": format_float(selected.minimum_radius_ratio),
        "geometry_min_radius_pair": selected.minimum_radius_pair,
        "geometry_min_li_li_distance_angstrom": format_float(
            selected.minimum_li_li_distance
        ),
        "geometry_min_o_o_distance_angstrom": format_float(
            selected.minimum_o_o_distance
        ),
        "geometry_min_o_f_distance_angstrom": format_float(
            selected.minimum_o_f_distance
        ),
        "geometry_issues": issue_text,
    }


def summarize_existing_row(row: dict[str, str]) -> str:
    status = row.get("ordering_batch_status", "")
    valid_count = row.get("geometry_valid_candidate_count", "")
    candidate_count = row.get("ordered_candidate_count", "")
    counts = (
        f"{valid_count}/{candidate_count} 个候选"
        if valid_count and candidate_count
        else "候选数未记录"
    )
    if status == "geometry_ok":
        return f"成功有序化（{counts}通过几何检查）"
    if status == "source_ordered":
        return "原始 CIF 已完全有序，并通过几何检查"
    if status == "manual_review":
        reason = row.get("geometry_issues", "") or "几何检查未通过"
        return f"未成功，需人工处理（{counts}通过；原因：{reason}）"
    if status == "ordering_error":
        return "有序化生成失败，需人工处理"
    if status:
        return f"未成功或待确认（状态：{status}）"
    return "未纳入有序化审计"


def merge_inventory(
    inventory_path: Path,
    updates: dict[str, dict[str, str]],
) -> None:
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    for field in AUDIT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in rows:
        row.update(updates.get(row.get("material_id", ""), {}))
        if not row.get("ordering_result_summary"):
            row["ordering_result_summary"] = summarize_existing_row(row)

    temporary_path = inventory_path.with_suffix(inventory_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(inventory_path)


def prune_inventory_outputs(
    inventory_path: Path,
    promote_dir: Path,
    error_dir: Path | None,
) -> None:
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        material_id = row.get("material_id", "")
        if not material_id:
            continue
        valid_count = int(row.get("geometry_valid_candidate_count", "") or 0)
        error_count = int(row.get("geometry_error_candidate_count", "") or 0)
        expected_valid = {
            output_path(promote_dir, material_id, index).resolve()
            for index in range(valid_count)
        }
        for existing in promote_dir.glob(f"{material_id}_ordered*.cif"):
            if existing.resolve() not in expected_valid:
                existing.unlink()
        if error_dir is None:
            continue
        expected_errors = {
            output_path(error_dir, material_id, index).resolve()
            for index in range(error_count)
        }
        for existing in error_dir.glob(f"{material_id}_ordered*.cif"):
            if existing.resolve() not in expected_errors:
                existing.unlink()


def main() -> int:
    args = build_parser().parse_args()
    args.promote_dir.mkdir(parents=True, exist_ok=True)
    if args.error_dir is not None:
        args.error_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, str]] = []
    for report_path in args.ordering_report:
        with report_path.open(encoding="utf-8", newline="") as stream:
            report_rows.extend(csv.DictReader(stream))

    by_id: dict[str, list[CandidateAudit]] = {}
    for row in report_rows:
        audit = audit_candidate(row, args)
        by_id.setdefault(audit.material_id, []).append(audit)

    updates: dict[str, dict[str, str]] = {}
    for material_id, candidates in by_id.items():
        errors = promote_error_candidates(candidates, args.error_dir)
        promoted = promote_candidates(candidates, args.promote_dir)
        updates[material_id] = audit_values(candidates, promoted, errors)
    merge_inventory(args.inventory, updates)
    prune_inventory_outputs(args.inventory, args.promote_dir, args.error_dir)

    valid_ids = sum(
        any(candidate.is_valid for candidate in candidates)
        for candidates in by_id.values()
    )
    print(
        f"Audited {len(by_id)} materials: geometry_ok={valid_ids} "
        f"manual_review={len(by_id) - valid_ids}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
