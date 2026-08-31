#!/usr/bin/env python3
"""Generate and optionally rank ordered approximants for disordered CIFs.

The script converts fractional occupancies into a bounded set of integer
occupations. It intentionally approximates refinement occupancies instead of
building arbitrarily large exact supercells. Optional CHGNet or MatterSim
relaxation ranks candidate orderings by predicted energy. Reference formulas
from all.csv protect trace elements and bound ordering-induced composition drift.
Multi-position split sites are treated as mutually exclusive local clusters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
  
VACANCY = "__vacancy__"
GROUP_PROPERTY = "_ordering_group"
CLUSTER_INSTANCE_PROPERTY = "_split_cluster_instance"
CLUSTER_ROLE_PROPERTY = "_split_cluster_role"
CLUSTER_OCC_PROPERTY = "_split_cluster_occupancy"

# Numerical tolerance for treating a site's summed occupancy as exactly one.
# This is a floating-point comparison guard, not a physical threshold, so it is a
# module constant rather than a CLI knob.
OCCUPANCY_SUM_TOLERANCE = 1e-5


@dataclass(frozen=True)
class SplitSitePair:
    species: str
    site_indices: tuple[int, int]
    distance: float
    occupancy_sum: float


@dataclass(frozen=True)
class SplitSiteCluster:
    family_id: int
    species: str
    site_indices: tuple[int, ...]
    max_center_distance: float
    occupancy_sum: float
    per_site_occupancies: tuple[float, ...]

    @property
    def cluster_size(self) -> int:
        return len(self.site_indices)


@dataclass(frozen=True)
class OccupancyGroup:
    group_id: int
    site_indices: tuple[int, ...]
    probabilities: dict[str, float]
    split_site_species: str | None = None
    cluster_size: int | None = None
    cluster_max_distance: float | None = None
    cluster_center_occupancy: float | None = None
    cluster_satellite_occupancy: float | None = None

    @property
    def is_disordered(self) -> bool:
        positive = [value for value in self.probabilities.values() if value > 0]
        return len(positive) > 1


@dataclass(frozen=True)
class AllocationPlan:
    multiplier: int
    scaling_matrix: tuple[int, int, int]
    allocations: dict[int, dict[str, int]]
    split_site_group_ids: tuple[int, ...]
    cluster_group_ids: tuple[int, ...]
    cluster_center_counts: dict[int, int]
    max_occupancy_error: float
    mean_occupancy_error: float
    composition_max_error: float | None
    composition_max_drift: float | None
    estimated_num_atoms: int
    within_error_tolerance: bool
    within_composition_tolerance: bool | None


@dataclass
class CandidateResult:
    parent_id: str
    source_file: str
    status: str
    attempt: int = 0
    message: str = ""
    reference_formula: str = ""
    source_formula: str = ""
    cleaned_formula: str = ""
    source_num_sites: int | None = None
    dropped_species_count: int = 0
    dropped_occupancy_sum: float = 0.0
    grouping_method: str = ""
    candidate_method: str = ""
    split_site_pair_count: int = 0
    split_site_cluster_count: int = 0
    split_site_species: str = ""
    supercell_multiplier: int | None = None
    supercell_matrix: str = ""
    occupancy_max_error: float | None = None
    occupancy_mean_error: float | None = None
    within_error_tolerance: bool | None = None
    source_composition_max_error: float | None = None
    composition_max_error: float | None = None
    composition_max_drift: float | None = None
    within_composition_tolerance: bool | None = None
    missing_reference_elements: str = ""
    candidate_index: int | None = None
    energy_rank: int | None = None
    num_atoms: int | None = None
    ordered_formula: str = ""
    ranker: str = "none"
    relaxed: bool = False
    converged: bool | None = None
    relaxation_steps: int | None = None
    relaxation_seconds: float | None = None
    total_energy_ev: float | None = None
    energy_per_atom_ev: float | None = None
    output_file: str = ""


@dataclass
class AnomalyResult:
    parent_id: str
    source_file: str
    stage: str
    status: str
    attempt: int = 0
    resolved: bool = False
    resolution: str = ""
    message: str = ""
    candidate_index: int | None = None
    num_atoms: int | None = None
    converged: bool | None = None
    relaxation_steps: int | None = None
    relaxation_seconds: float | None = None
    total_energy_ev: float | None = None
    energy_per_atom_ev: float | None = None


@dataclass(frozen=True)
class RankedStructure:
    structure: Structure
    relaxed: bool
    total_energy_ev: float | None
    energy_per_atom_ev: float | None
    error: str = ""
    converged: bool | None = None
    relaxation_steps: int | None = None
    relaxation_seconds: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ordered approximants from fractional-occupancy CIFs and "
            "optionally rank candidates with an ML interatomic potential."
        )
    )

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument("--input-dir", type=Path, default=Path("OBELIX/cifs"))
    io_group.add_argument(
        "--output-dir", type=Path, default=Path("OBELIX/ordered_cifs")
    )
    io_group.add_argument(
        "--report-path", type=Path, default=Path("OBELIX/ordering_report.csv")
    )
    io_group.add_argument(
        "--anomaly-path",
        type=Path,
        default=None,
        help=(
            "CSV for processing, ordering, and relaxation anomalies. By default, "
            "place it next to --report-path with an _anomalies suffix."
        ),
    )
    io_group.add_argument(
        "--composition-csv",
        type=Path,
        default=Path("OBELIX/all.csv"),
        help=(
            "CSV containing ID and True Composition columns used as a soft "
            "stoichiometry reference."
        ),
    )
    io_group.add_argument(
        "--include-ordered",
        action="store_true",
        help="Copy already ordered inputs into the output dataset.",
    )
    io_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace colliding output files and the existing report.",
    )
    io_group.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing reports and skip every material already in the main report.",
    )
    io_group.add_argument(
        "--retry-anomalies",
        action="store_true",
        help=(
            "Reprocess only materials with retryable errors in --anomaly-path. "
            "A new deterministic random seed is used for each retry attempt."
        ),
    )
    io_group.add_argument("--progress-every", type=int, default=1)

    select_group = parser.add_argument_group("input selection")
    select_group.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated material IDs. By default, scan every CIF.",
    )
    select_group.add_argument(
        "--start-after",
        type=str,
        default=None,
        help="Resume the sorted input list after this material ID.",
    )
    select_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N inputs after selection. For test runs.",
    )

    composition_group = parser.add_argument_group("occupancy cleaning & composition")
    composition_group.add_argument(
        "--min-occupancy",
        type=float,
        default=0.01,
        help="Drop species with occupancy strictly below this threshold.",
    )
    composition_group.add_argument(
        "--max-occupancy-error",
        type=float,
        default=0.05,
        help="Maximum allowed per-category occupancy approximation error.",
    )
    composition_group.add_argument(
        "--composition-tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum added atomic-fraction error relative to all.csv. Existing "
            "CIF-to-CSV differences are treated as the baseline."
        ),
    )
    composition_group.add_argument(
        "--preserve-retained-species",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep at least one atom for every species above --min-occupancy.",
    )

    split_group = parser.add_argument_group("split-site detection")
    split_group.add_argument(
        "--detect-split-sites",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat close partial sites of the same element as mutually exclusive pairs.",
    )
    split_group.add_argument(
        "--split-site-max-distance",
        type=float,
        default=1.2,
        help="Largest separation in angstrom for detecting a split-site pair.",
    )
    split_group.add_argument(
        "--split-site-occupancy-tolerance",
        type=float,
        default=0.1,
        help="Allowed deviation of a split-site pair's summed occupancy from one.",
    )
    split_group.add_argument(
        "--detect-split-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat symmetry-related multi-position split sites as local clusters.",
    )
    split_group.add_argument(
        "--split-cluster-max-distance",
        type=float,
        default=1.5,
        help="Largest center-to-satellite distance for a split-site cluster.",
    )
    split_group.add_argument(
        "--split-cluster-occupancy-tolerance",
        type=float,
        default=0.1,
        help="Allowed deviation of a cluster's summed occupancy from one.",
    )

    candidate_group = parser.add_argument_group("supercell & candidate generation")
    candidate_group.add_argument(
        "--max-supercell-multiplier",
        type=int,
        default=8,
        help="Largest determinant considered for a diagonal supercell.",
    )
    candidate_group.add_argument(
        "--max-atoms",
        type=int,
        default=128,
        help="Maximum number of occupied atoms in an ordered candidate.",
    )
    candidate_group.add_argument("--num-candidates", type=int, default=5)
    candidate_group.add_argument(
        "--candidate-method",
        choices=("random", "enumerate", "ewald", "auto"),
        default="random",
        help=(
            "How to generate distinct orderings for non-split-site structures. "
            "'random': legacy random assignment + StructureMatcher dedup. "
            "'enumerate': exact enumlib enumeration (needs enum.x/makestr.x on PATH). "
            "'ewald': pymatgen Ewald-ranked ordering, no external binary. "
            "'auto': try enumerate, then ewald, then random. Structures with "
            "split-site pairs or clusters always fall back to random."
        ),
    )
    candidate_group.add_argument(
        "--ewald-algo",
        choices=("fast", "complete", "best_first"),
        default="fast",
        help=(
            "Enumeration algorithm for --candidate-method ewald. 'fast': a few "
            "low-energy orderings (may return fewer than --num-candidates). "
            "'complete': every symmetry-distinct ordering (slow for many "
            "disordered sites). 'best_first': greedy lowest-energy first. "
            "Ignored for other candidate methods."
        ),
    )
    candidate_group.add_argument(
        "--max-attempts-per-candidate",
        type=int,
        default=20,
        help="Random attempts used to obtain each symmetry-distinct candidate.",
    )
    candidate_group.add_argument("--symprec", type=float, default=0.1)
    candidate_group.add_argument("--seed", type=int, default=17)

    ranker_group = parser.add_argument_group(
        "MLIP ranking", "Only used when --ranker is chgnet or mattersim."
    )
    ranker_group.add_argument(
        "--ranker", choices=("none", "chgnet", "mattersim"), default="none"
    )
    ranker_group.add_argument(
        "--potential-path",
        type=Path,
        default=None,
        help="Optional local MatterSim checkpoint. Avoids an automatic download.",
    )
    ranker_group.add_argument("--device", type=str, default="cpu")
    ranker_group.add_argument("--fmax", type=float, default=0.08)
    ranker_group.add_argument(
        "--relax-steps",
        type=int,
        default=300,
        help="Maximum optimization steps per candidate for CHGNet and MatterSim.",
    )
    ranker_group.add_argument(
        "--relax-timeout",
        type=float,
        default=600.0,
        help=(
            "Maximum MatterSim wall time in seconds per active candidate. "
            "Zero disables the time limit."
        ),
    )
    ranker_group.add_argument(
        "--step-progress-every",
        type=int,
        default=25,
        help="Print MatterSim step distributions every N batch optimization steps.",
    )
    ranker_group.add_argument(
        "--relax-cell", action=argparse.BooleanOptionalAction, default=True
    )
    ranker_group.add_argument("--max-natoms-per-batch", type=int, default=512)
    ranker_group.add_argument(
        "--keep-top",
        type=int,
        default=0,
        help="Keep only the N lowest-energy candidates. Zero keeps all.",
    )
    return parser


def stable_seed(base_seed: int, material_id: str) -> int:
    digest = hashlib.sha256(material_id.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def load_reference_compositions(path: Path) -> dict[str, Composition]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"ID", "True Composition"}
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"Missing columns in {path}: {', '.join(sorted(missing_columns))}"
            )

        references: dict[str, Composition] = {}
        for line_number, row in enumerate(reader, start=2):
            material_id = row["ID"].strip()
            formula = row["True Composition"].strip()
            if not material_id or not formula:
                continue
            if material_id in references:
                raise ValueError(f"Duplicate material ID {material_id!r} on line {line_number}")
            try:
                references[material_id] = Composition(formula)
            except Exception as exc:
                raise ValueError(
                    f"Invalid True Composition for {material_id!r} on line "
                    f"{line_number}: {formula!r}"
                ) from exc
    return references


def atomic_fractions(composition: Composition | Mapping[str, float]) -> dict[str, float]:
    amounts: dict[str, float] = {}
    for element, amount in composition.items():
        if float(amount) <= 0:
            continue
        symbol = element.symbol if hasattr(element, "symbol") else str(element)
        amounts[symbol] = amounts.get(symbol, 0.0) + float(amount)
    total = sum(amounts.values())
    if total <= 0:
        raise ValueError("Cannot compare an empty composition")
    return {element: amount / total for element, amount in amounts.items()}


def composition_errors(
    reference: Composition,
    actual: Composition | Mapping[str, float],
) -> dict[str, float]:
    reference_fractions = atomic_fractions(reference)
    actual_fractions = atomic_fractions(actual)
    elements = reference_fractions.keys() | actual_fractions.keys()
    return {
        element: abs(
            actual_fractions.get(element, 0.0) - reference_fractions.get(element, 0.0)
        )
        for element in elements
    }


def composition_error_metrics(
    reference: Composition,
    actual: Composition | Mapping[str, float],
    baseline: Composition | Mapping[str, float],
) -> tuple[float, float]:
    actual_errors = composition_errors(reference, actual)
    baseline_errors = composition_errors(reference, baseline)
    elements = actual_errors.keys() | baseline_errors.keys()
    max_error = max(actual_errors.values(), default=0.0)
    max_drift = max(
        (
            max(
                0.0,
                actual_errors.get(element, 0.0)
                - baseline_errors.get(element, 0.0),
            )
            for element in elements
        ),
        default=0.0,
    )
    return max_error, max_drift


def normalize_probabilities(
    probabilities: dict[str, float], tolerance: float
) -> dict[str, float]:
    total = sum(probabilities.values())
    if total > 1.0 + tolerance:
        raise ValueError(f"Site occupancy sum {total:.6f} exceeds one")
    if math.isclose(total, 1.0, abs_tol=tolerance):
        return {key: value / total for key, value in probabilities.items()}
    normalized = dict(probabilities)
    normalized[VACANCY] = max(0.0, 1.0 - total)
    return normalized


def clean_small_occupancies(
    structure: Structure,
    min_occupancy: float,
    occupancy_sum_tolerance: float = OCCUPANCY_SUM_TOLERANCE,
    protected_elements: Collection[str] = (),
) -> tuple[Structure, int, float]:
    cleaned = structure.copy()
    remove_indices: list[int] = []
    dropped_species_count = 0
    dropped_occupancy_sum = 0.0
    protected = set(protected_elements)
    source_elements = {
        species.symbol
        for site in structure
        for species, occupancy in site.species.items()
        if occupancy > 0
    }
    elements_above_threshold = {
        species.symbol
        for site in structure
        for species, occupancy in site.species.items()
        if occupancy >= min_occupancy
    }
    keep_below_threshold = (protected & source_elements) - elements_above_threshold

    for index, site in enumerate(structure):
        original = {species: float(occupancy) for species, occupancy in site.species.items()}
        kept: dict[Any, float] = {}
        for species, occupancy in original.items():
            if occupancy < min_occupancy and species.symbol not in keep_below_threshold:
                dropped_species_count += 1
                dropped_occupancy_sum += occupancy
            else:
                kept[species] = occupancy

        if not kept:
            remove_indices.append(index)
            continue

        original_total = sum(original.values())
        if math.isclose(original_total, 1.0, abs_tol=occupancy_sum_tolerance):
            kept_total = sum(kept.values())
            kept = {species: value / kept_total for species, value in kept.items()}
        cleaned.replace(index, Composition(kept))

    if remove_indices:
        cleaned.remove_sites(remove_indices)
    return cleaned, dropped_species_count, dropped_occupancy_sum


def species_signature(site: Any) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (species.symbol, round(float(occupancy), 8))
            for species, occupancy in site.species.items()
        )
    )


def find_split_site_clusters(
    structure: Structure,
    max_distance: float,
    occupancy_tolerance: float,
    symprec: float = 0.1,
) -> list[SplitSiteCluster]:
    """Find one-atom clusters formed by two symmetry orbits."""
    try:
        equivalent_indices = SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_symmetrized_structure().equivalent_indices
    except Exception:
        return []

    orbit_data: dict[str, list[tuple[tuple[int, ...], float]]] = {}
    for indices in equivalent_indices:
        orbit = tuple(int(index) for index in indices)
        reference = structure[orbit[0]]
        if len(reference.species) != 1:
            continue
        species, occupancy = next(iter(reference.species.items()))
        value = float(occupancy)
        if 0.0 < value < 1.0:
            orbit_data.setdefault(species.symbol, []).append((orbit, value))

    clusters: list[SplitSiteCluster] = []
    family_id = 0
    for species, element_orbits in orbit_data.items():
        sorted_orbits = sorted(element_orbits, key=lambda item: len(item[0]))
        used_orbits: set[int] = set()
        for center_position, (center_indices, center_occupancy) in enumerate(
            sorted_orbits
        ):
            if center_position in used_orbits:
                continue
            for satellite_position, (
                satellite_indices,
                satellite_occupancy,
            ) in enumerate(sorted_orbits):
                if satellite_position == center_position:
                    continue
                if satellite_position in used_orbits:
                    continue
                if len(satellite_indices) % len(center_indices):
                    continue
                satellites_per_center = len(satellite_indices) // len(center_indices)
                if satellites_per_center < 2:
                    continue

                occupancy_sum = (
                    center_occupancy
                    + satellites_per_center * satellite_occupancy
                )
                if abs(occupancy_sum - 1.0) > occupancy_tolerance + 1e-12:
                    continue

                instances: list[tuple[int, tuple[int, ...], float]] = []
                satellite_usage: dict[int, int] = {
                    index: 0 for index in satellite_indices
                }
                valid = True
                for center_index in center_indices:
                    neighbors = sorted(
                        (
                            float(structure.get_distance(center_index, satellite_index)),
                            satellite_index,
                        )
                        for satellite_index in satellite_indices
                        if structure.get_distance(center_index, satellite_index)
                        <= max_distance + 1e-8
                    )
                    if len(neighbors) != satellites_per_center:
                        valid = False
                        break
                    for _, satellite_index in neighbors:
                        satellite_usage[satellite_index] += 1
                    instances.append(
                        (
                            center_index,
                            tuple(index for _, index in neighbors),
                            max(distance for distance, _ in neighbors),
                        )
                    )

                if not valid or any(count != 1 for count in satellite_usage.values()):
                    continue

                for center_index, nearby_satellites, farthest in instances:
                    site_indices = (center_index, *nearby_satellites)
                    clusters.append(
                        SplitSiteCluster(
                            family_id=family_id,
                            species=species,
                            site_indices=site_indices,
                            max_center_distance=farthest,
                            occupancy_sum=occupancy_sum,
                            per_site_occupancies=(
                                center_occupancy,
                                *(satellite_occupancy for _ in nearby_satellites),
                            ),
                        )
                    )
                used_orbits.update((center_position, satellite_position))
                family_id += 1
                break

    return clusters


def find_split_site_pairs(
    structure: Structure,
    max_distance: float,
    occupancy_tolerance: float,
    excluded_indices: Collection[int] = (),
) -> list[SplitSitePair]:
    excluded = set(excluded_indices)
    partial_sites: dict[str, list[tuple[int, float]]] = {}
    for index, site in enumerate(structure):
        if index in excluded:
            continue
        if len(site.species) != 1:
            continue
        species, occupancy = next(iter(site.species.items()))
        value = float(occupancy)
        if 0.0 < value < 1.0:
            partial_sites.setdefault(species.symbol, []).append((index, value))

    if not partial_sites:
        return []

    distances = np.asarray(structure.distance_matrix, dtype=float)
    candidates: list[tuple[float, str, int, int, float]] = []
    for species, sites in partial_sites.items():
        if len(sites) < 2:
            continue
        nearest = {
            index: min(
                distances[index, other_index]
                for other_index, _ in sites
                if other_index != index
            )
            for index, _ in sites
        }
        for (first_index, first_occupancy), (
            second_index,
            second_occupancy,
        ) in itertools.combinations(sites, 2):
            distance = float(distances[first_index, second_index])
            occupancy_sum = first_occupancy + second_occupancy
            if distance > max_distance:
                continue
            if abs(occupancy_sum - 1.0) > occupancy_tolerance + 1e-12:
                continue
            if distance > nearest[first_index] + 1e-6:
                continue
            if distance > nearest[second_index] + 1e-6:
                continue
            candidates.append(
                (distance, species, first_index, second_index, occupancy_sum)
            )

    pairs: list[SplitSitePair] = []
    assigned_indices: set[int] = set()
    for distance, species, first_index, second_index, occupancy_sum in sorted(candidates):
        if first_index in assigned_indices or second_index in assigned_indices:
            continue
        pairs.append(
            SplitSitePair(
                species=species,
                site_indices=(first_index, second_index),
                distance=distance,
                occupancy_sum=occupancy_sum,
            )
        )
        assigned_indices.update((first_index, second_index))
    return pairs


def find_occupancy_groups(
    structure: Structure,
    symprec: float,
    occupancy_sum_tolerance: float = OCCUPANCY_SUM_TOLERANCE,
    split_site_pairs: Sequence[SplitSitePair] = (),
    split_site_clusters: Sequence[SplitSiteCluster] = (),
) -> tuple[list[OccupancyGroup], str]:
    try:
        equivalent_indices = SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_symmetrized_structure().equivalent_indices
        grouping_method = "spacegroup"
    except Exception:
        by_signature: dict[tuple[tuple[str, float], ...], list[int]] = {}
        for index, site in enumerate(structure):
            by_signature.setdefault(species_signature(site), []).append(index)
        equivalent_indices = list(by_signature.values())
        grouping_method = "species_signature_fallback"

    assigned_split_indices: set[int] = set()
    clusters_by_family: dict[int, list[SplitSiteCluster]] = {}
    for cluster in split_site_clusters:
        overlap = assigned_split_indices & set(cluster.site_indices)
        if overlap:
            raise ValueError(
                f"Split-site clusters overlap at site indices {sorted(overlap)}"
            )
        assigned_split_indices.update(cluster.site_indices)
        clusters_by_family.setdefault(cluster.family_id, []).append(cluster)

    for pair in split_site_pairs:
        overlap = assigned_split_indices & set(pair.site_indices)
        if overlap:
            raise ValueError(
                f"Split-site pairs overlap at site indices {sorted(overlap)}"
            )
        assigned_split_indices.update(pair.site_indices)

    grouped_indices: list[
        tuple[
            tuple[int, ...],
            SplitSitePair | None,
            list[SplitSiteCluster] | None,
        ]
    ] = []
    for family_clusters in clusters_by_family.values():
        family_indices = tuple(
            index
            for cluster in family_clusters
            for index in cluster.site_indices
        )
        grouped_indices.append((family_indices, None, family_clusters))
    for pair in split_site_pairs:
        grouped_indices.append((pair.site_indices, pair, None))
    for indices in equivalent_indices:
        remaining = tuple(
            int(index) for index in indices if index not in assigned_split_indices
        )
        if remaining:
            grouped_indices.append((remaining, None, None))

    if split_site_clusters:
        grouping_method += "+split_site_clusters"
    if split_site_pairs:
        grouping_method += "+split_site_pairs"

    groups: list[OccupancyGroup] = []
    group_ids = [-1] * len(structure)
    for group_id, (indices, split_pair, family_clusters) in enumerate(grouped_indices):
        if family_clusters is not None:
            reference_cluster = family_clusters[0]
            cluster_size = reference_cluster.cluster_size
            center_occupancy = reference_cluster.per_site_occupancies[0]
            satellite_occupancy = reference_cluster.per_site_occupancies[1]
            if any(cluster.cluster_size != cluster_size for cluster in family_clusters):
                raise ValueError(f"Inconsistent split-site cluster family {group_id}")
            average_occupancy = reference_cluster.occupancy_sum / cluster_size
            probabilities = {
                reference_cluster.species: average_occupancy,
                VACANCY: 1.0 - average_occupancy,
            }
            group = OccupancyGroup(
                group_id=group_id,
                site_indices=indices,
                probabilities=probabilities,
                split_site_species=reference_cluster.species,
                cluster_size=cluster_size,
                cluster_max_distance=max(
                    cluster.max_center_distance for cluster in family_clusters
                ),
                cluster_center_occupancy=center_occupancy,
                cluster_satellite_occupancy=satellite_occupancy,
            )
        elif split_pair is None:
            reference = structure[indices[0]]
            probabilities = {
                species.symbol: float(occupancy)
                for species, occupancy in reference.species.items()
            }
            probabilities = normalize_probabilities(
                probabilities, occupancy_sum_tolerance
            )
            group = OccupancyGroup(
                group_id=group_id,
                site_indices=indices,
                probabilities=probabilities,
            )
        else:
            probabilities = {
                split_pair.species: split_pair.occupancy_sum / len(indices)
            }
            probabilities = normalize_probabilities(
                probabilities, occupancy_sum_tolerance
            )
            group = OccupancyGroup(
                group_id=group_id,
                site_indices=indices,
                probabilities=probabilities,
                split_site_species=split_pair.species,
            )
        groups.append(group)
        for index in indices:
            group_ids[index] = group_id

    if any(group_id < 0 for group_id in group_ids):
        raise RuntimeError("Some sites were not assigned to an occupancy group")
    structure.add_site_property(GROUP_PROPERTY, group_ids)

    cluster_instance_ids = [-1] * len(structure)
    cluster_roles = [-1] * len(structure)
    cluster_occupancies = [0.0] * len(structure)
    for instance_id, cluster in enumerate(split_site_clusters):
        for role, (index, occupancy) in enumerate(
            zip(cluster.site_indices, cluster.per_site_occupancies)
        ):
            cluster_instance_ids[index] = instance_id
            cluster_roles[index] = 0 if role == 0 else 1
            cluster_occupancies[index] = occupancy
    structure.add_site_property(CLUSTER_INSTANCE_PROPERTY, cluster_instance_ids)
    structure.add_site_property(CLUSTER_ROLE_PROPERTY, cluster_roles)
    structure.add_site_property(CLUSTER_OCC_PROPERTY, cluster_occupancies)
    return groups, grouping_method


def largest_remainder_allocation(
    num_positions: int,
    probabilities: dict[str, float],
) -> tuple[dict[str, int], list[float]]:
    raw = {key: num_positions * value for key, value in probabilities.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = num_positions - sum(counts.values())
    ranked = sorted(
        probabilities,
        key=lambda key: (raw[key] - counts[key], probabilities[key], key),
        reverse=True,
    )
    for key in ranked[:remainder]:
        counts[key] += 1

    errors = [abs(counts[key] / num_positions - probabilities[key]) for key in probabilities]
    return counts, errors


def preserve_species_globally(
    groups: Sequence[OccupancyGroup],
    allocations: dict[int, dict[str, int]],
    multiplier: int,
    required_species: Collection[str],
) -> None:
    required = set(required_species) - {VACANCY}
    global_counts: dict[str, int] = {}
    for counts in allocations.values():
        for species, count in counts.items():
            global_counts[species] = global_counts.get(species, 0) + count

    missing = {species for species in required if global_counts.get(species, 0) == 0}
    while missing:
        recipient = min(
            missing,
            key=lambda species: (
                sum(species in group.probabilities for group in groups),
                species,
            ),
        )
        transfers: list[tuple[float, float, int, str]] = []
        for group in groups:
            if group.probabilities.get(recipient, 0.0) <= 0:
                continue
            counts = allocations[group.group_id]
            num_positions = len(group.site_indices) * multiplier
            for donor, donor_count in counts.items():
                if donor == recipient or donor_count <= 0:
                    continue
                if donor in required and global_counts.get(donor, 0) <= 1:
                    continue
                before = abs(
                    counts[donor] / num_positions - group.probabilities[donor]
                )
                before += abs(
                    counts[recipient] / num_positions - group.probabilities[recipient]
                )
                after = abs(
                    (counts[donor] - 1) / num_positions - group.probabilities[donor]
                )
                after += abs(
                    (counts[recipient] + 1) / num_positions
                    - group.probabilities[recipient]
                )
                transfers.append(
                    (
                        after - before,
                        -group.probabilities[recipient],
                        group.group_id,
                        donor,
                    )
                )

        if not transfers:
            raise ValueError(
                f"No allocation can preserve required species {recipient!r} "
                f"at multiplier {multiplier}"
            )
        _, _, group_id, donor = min(transfers)
        allocations[group_id][donor] -= 1
        allocations[group_id][recipient] += 1
        global_counts[donor] -= 1
        global_counts[recipient] = 1
        missing.remove(recipient)


def allocation_counts(allocations: Mapping[int, Mapping[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group_counts in allocations.values():
        for species, count in group_counts.items():
            if species != VACANCY:
                counts[species] = counts.get(species, 0) + count
    return counts


def diagonal_scaling_matrix(structure: Structure, multiplier: int) -> tuple[int, int, int]:
    triples: set[tuple[int, int, int]] = set()
    for a in range(1, multiplier + 1):
        if multiplier % a:
            continue
        quotient = multiplier // a
        for b in range(1, quotient + 1):
            if quotient % b:
                continue
            c = quotient // b
            triples.update(itertools.permutations((a, b, c)))

    lengths = np.asarray(structure.lattice.abc, dtype=float)

    def shape_score(triple: tuple[int, int, int]) -> tuple[float, float]:
        scaled = lengths * np.asarray(triple)
        return float(scaled.max() / scaled.min()), float(scaled.max())

    return min(triples, key=shape_score)


def build_allocation_plan(
    structure: Structure,
    groups: Sequence[OccupancyGroup],
    max_multiplier: int,
    max_atoms: int,
    max_occupancy_error: float,
    preserve_retained_species: bool,
    reference_composition: Composition | None = None,
    baseline_composition: Composition | None = None,
    composition_tolerance: float = 0.05,
    required_species: Collection[str] = (),
) -> AllocationPlan:
    feasible: list[AllocationPlan] = []
    retained_species = {
        species
        for group in groups
        for species, probability in group.probabilities.items()
        if species != VACANCY and probability > 0
    }
    species_to_preserve = set(required_species)
    if preserve_retained_species:
        species_to_preserve.update(retained_species)

    for multiplier in range(1, max_multiplier + 1):
        allocations: dict[int, dict[str, int]] = {}
        cluster_center_counts: dict[int, int] = {}
        for group in groups:
            num_positions = len(group.site_indices) * multiplier
            if group.cluster_size is not None:
                num_instances = len(group.site_indices) // group.cluster_size
                total_clusters = num_instances * multiplier
                species = group.split_site_species
                if species is None:
                    raise ValueError(
                        f"Split-site cluster group {group.group_id} has no species"
                    )
                counts = {
                    species: total_clusters,
                    VACANCY: total_clusters * (group.cluster_size - 1),
                }
                center_occupancy = group.cluster_center_occupancy or 0.0
                satellite_occupancy = group.cluster_satellite_occupancy or 0.0
                normalized_center = center_occupancy / (
                    center_occupancy
                    + (group.cluster_size - 1) * satellite_occupancy
                )
                role_counts, _ = largest_remainder_allocation(
                    total_clusters,
                    {"center": normalized_center, "satellite": 1.0 - normalized_center},
                )
                cluster_center_counts[group.group_id] = role_counts["center"]
            elif group.split_site_species is not None:
                if len(group.site_indices) != 2:
                    raise ValueError(
                        f"Split-site group {group.group_id} must contain two sites"
                    )
                counts = {
                    group.split_site_species: multiplier,
                    VACANCY: multiplier,
                }
            else:
                counts, _ = largest_remainder_allocation(
                    num_positions,
                    group.probabilities,
                )
            allocations[group.group_id] = counts

        try:
            preserve_species_globally(
                groups,
                allocations,
                multiplier,
                required_species=species_to_preserve,
            )
        except ValueError:
            continue

        errors: list[float] = []
        for group in groups:
            if group.cluster_size is not None:
                total_clusters = (
                    len(group.site_indices) // group.cluster_size * multiplier
                )
                center_count = cluster_center_counts[group.group_id]
                satellite_count = total_clusters - center_count
                errors.extend(
                    (
                        abs(
                            center_count / total_clusters
                            - (group.cluster_center_occupancy or 0.0)
                        ),
                        abs(
                            satellite_count
                            / ((group.cluster_size - 1) * total_clusters)
                            - (group.cluster_satellite_occupancy or 0.0)
                        ),
                    )
                )
                continue
            num_positions = len(group.site_indices) * multiplier
            errors.extend(
                abs(
                    allocations[group.group_id].get(species, 0) / num_positions
                    - probability
                )
                for species, probability in group.probabilities.items()
            )
        ordered_counts = allocation_counts(allocations)
        estimated_num_atoms = sum(ordered_counts.values())

        if estimated_num_atoms > max_atoms:
            continue
        max_error = max(errors, default=0.0)
        composition_max_error: float | None = None
        composition_max_drift: float | None = None
        within_composition_tolerance: bool | None = None
        if reference_composition is not None:
            baseline = baseline_composition or structure.composition
            composition_max_error, composition_max_drift = composition_error_metrics(
                reference_composition,
                ordered_counts,
                baseline,
            )
            within_composition_tolerance = (
                composition_max_drift <= composition_tolerance + 1e-12
            )
        plan = AllocationPlan(
            multiplier=multiplier,
            scaling_matrix=diagonal_scaling_matrix(structure, multiplier),
            allocations=allocations,
            split_site_group_ids=tuple(
                group.group_id
                for group in groups
                if group.split_site_species is not None and group.cluster_size is None
            ),
            cluster_group_ids=tuple(
                group.group_id for group in groups if group.cluster_size is not None
            ),
            cluster_center_counts=cluster_center_counts,
            max_occupancy_error=max_error,
            mean_occupancy_error=float(np.mean(errors)) if errors else 0.0,
            composition_max_error=composition_max_error,
            composition_max_drift=composition_max_drift,
            estimated_num_atoms=estimated_num_atoms,
            within_error_tolerance=max_error <= max_occupancy_error + 1e-12,
            within_composition_tolerance=within_composition_tolerance,
        )
        feasible.append(plan)
        if plan.within_error_tolerance and plan.within_composition_tolerance is not False:
            return plan

    if not feasible:
        raise ValueError(
            f"No supercell up to multiplier {max_multiplier} fits max_atoms={max_atoms}"
        )
    if reference_composition is None:
        return min(
            feasible,
            key=lambda plan: (
                plan.max_occupancy_error,
                plan.mean_occupancy_error,
                plan.multiplier,
            ),
        )

    def plan_score(plan: AllocationPlan) -> tuple[float, float, float, float, int]:
        occupancy_excess = max(0.0, plan.max_occupancy_error - max_occupancy_error)
        composition_drift = plan.composition_max_drift or 0.0
        composition_excess = max(0.0, composition_drift - composition_tolerance)
        return (
            max(occupancy_excess, composition_excess),
            occupancy_excess + composition_excess,
            composition_drift,
            plan.max_occupancy_error,
            plan.multiplier,
        )

    return min(feasible, key=plan_score)


def nearest_site_pairs(
    structure: Structure,
    indices: Sequence[int],
) -> list[tuple[int, int]]:
    if len(indices) % 2:
        raise ValueError(f"Cannot pair an odd number of split sites: {len(indices)}")
    edges = sorted(
        (
            structure.get_distance(first, second),
            first,
            second,
        )
        for first, second in itertools.combinations(indices, 2)
    )
    assigned: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, first, second in edges:
        if first in assigned or second in assigned:
            continue
        pairs.append((first, second))
        assigned.update((first, second))
    if len(assigned) != len(indices):
        raise RuntimeError("Some split sites could not be paired in the supercell")
    return pairs


def split_cluster_copies(
    structure: Structure,
    indices: Sequence[int],
    cluster_size: int,
    max_center_distance: float,
) -> list[tuple[int, ...]]:
    """Recover replicated cluster instances from persistent site labels."""
    by_instance: dict[int, list[int]] = {}
    for index in indices:
        instance_id = int(structure[index].properties[CLUSTER_INSTANCE_PROPERTY])
        by_instance.setdefault(instance_id, []).append(index)

    clusters: list[tuple[int, ...]] = []
    satellites_per_center = cluster_size - 1
    for instance_id, instance_indices in sorted(by_instance.items()):
        if instance_id < 0:
            raise RuntimeError("Split-site cluster is missing an instance label")
        centers = sorted(
            index
            for index in instance_indices
            if int(structure[index].properties[CLUSTER_ROLE_PROPERTY]) == 0
        )
        satellites = sorted(
            index
            for index in instance_indices
            if int(structure[index].properties[CLUSTER_ROLE_PROPERTY]) == 1
        )
        if len(satellites) != satellites_per_center * len(centers):
            raise RuntimeError(
                f"Cluster instance {instance_id} has inconsistent role counts"
            )

        satellite_usage = {index: 0 for index in satellites}
        instance_clusters: list[tuple[int, ...]] = []
        for center in centers:
            nearby = sorted(
                (
                    float(structure.get_distance(center, satellite)),
                    satellite,
                )
                for satellite in satellites
                if structure.get_distance(center, satellite)
                <= max_center_distance + 1e-5
            )
            if len(nearby) != satellites_per_center:
                raise RuntimeError(
                    f"Cluster instance {instance_id} cannot be reconstructed "
                    "unambiguously after supercell expansion"
                )
            for _, satellite in nearby:
                satellite_usage[satellite] += 1
            instance_clusters.append((center, *(index for _, index in nearby)))

        if any(count != 1 for count in satellite_usage.values()):
            raise RuntimeError(
                f"Cluster instance {instance_id} overlaps after supercell expansion"
            )
        clusters.extend(instance_clusters)
    return clusters


def make_ordered_candidate(
    decorated_structure: Structure,
    plan: AllocationPlan,
    groups: Sequence[OccupancyGroup],
    rng: random.Random,
) -> Structure:
    candidate = decorated_structure.copy()
    candidate.make_supercell(plan.scaling_matrix)

    indices_by_group: dict[int, list[int]] = {}
    for index, site in enumerate(candidate):
        group_id = int(site.properties[GROUP_PROPERTY])
        indices_by_group.setdefault(group_id, []).append(index)

    group_by_id = {group.group_id: group for group in groups}
    split_site_group_ids = set(plan.split_site_group_ids)
    cluster_group_ids = set(plan.cluster_group_ids)
    remove_indices: list[int] = []
    for group_id, indices in indices_by_group.items():
        if group_id in cluster_group_ids:
            group = group_by_id[group_id]
            cluster_size = group.cluster_size
            max_distance = group.cluster_max_distance
            species = group.split_site_species
            if cluster_size is None or max_distance is None or species is None:
                raise RuntimeError(f"Incomplete split-site cluster group {group_id}")
            clusters = split_cluster_copies(
                candidate,
                indices,
                cluster_size=cluster_size,
                max_center_distance=max_distance,
            )
            center_count = plan.cluster_center_counts[group_id]
            if not 0 <= center_count <= len(clusters):
                raise RuntimeError(f"Invalid center count for cluster group {group_id}")
            choose_center = [True] * center_count + [False] * (
                len(clusters) - center_count
            )
            rng.shuffle(choose_center)
            for cluster, use_center in zip(clusters, choose_center):
                chosen = cluster[0] if use_center else rng.choice(cluster[1:])
                candidate.replace(chosen, Element(species))
                remove_indices.extend(index for index in cluster if index != chosen)
            continue

        if group_id in split_site_group_ids:
            counts = plan.allocations[group_id]
            occupied_species = [
                species
                for species, count in counts.items()
                if species != VACANCY and count > 0
            ]
            pairs = nearest_site_pairs(candidate, indices)
            if len(occupied_species) != 1:
                raise RuntimeError(
                    f"Split-site group {group_id} must contain exactly one species"
                )
            species = occupied_species[0]
            if counts[species] != len(pairs) or counts.get(VACANCY, 0) != len(pairs):
                raise RuntimeError(
                    f"Split-site group {group_id} does not have one atom and one "
                    "vacancy per pair"
                )
            for first, second in pairs:
                occupied, vacant = (first, second) if rng.random() < 0.5 else (second, first)
                candidate.replace(occupied, Element(species))
                remove_indices.append(vacant)
            continue

        assignments: list[str] = []
        for species, count in plan.allocations[group_id].items():
            assignments.extend([species] * count)
        if len(assignments) != len(indices):
            raise RuntimeError(
                f"Group {group_id} has {len(indices)} sites but {len(assignments)} assignments"
            )
        rng.shuffle(assignments)
        for index, assignment in zip(sorted(indices), assignments):
            if assignment == VACANCY:
                remove_indices.append(index)
            else:
                candidate.replace(index, Element(assignment))

    if remove_indices:
        candidate.remove_sites(sorted(set(remove_indices), reverse=True))
    for property_name in (
        GROUP_PROPERTY,
        CLUSTER_INSTANCE_PROPERTY,
        CLUSTER_ROLE_PROPERTY,
        CLUSTER_OCC_PROPERTY,
    ):
        if property_name in candidate.site_properties:
            candidate.remove_site_property(property_name)
    if not candidate.is_ordered:
        raise RuntimeError("Candidate remains disordered after integer assignment")

    return finalize_candidate_cell(candidate)


def finalize_candidate_cell(structure: Structure) -> Structure:
    """Reduce a decorated candidate to its primitive, Niggli-reduced cell."""
    primitive = structure.get_primitive_structure(tolerance=0.1)
    return primitive.get_reduced_structure(reduction_algo="niggli")


def plan_occupancy_supercell(
    decorated_structure: Structure,
    plan: AllocationPlan,
) -> Structure:
    """Expand to the planned supercell and stamp each occupancy group's integer
    counts as equal per-site fractional occupancies.

    Because every site in a group is symmetry-equivalent, an equal fractional
    occupancy commensurate with the supercell reproduces exactly the plan's atom
    counts, which an exact enumerator then arranges in every distinct way.
    """
    supercell = decorated_structure.copy()
    supercell.make_supercell(plan.scaling_matrix)
    indices_by_group: dict[int, list[int]] = {}
    for index, site in enumerate(supercell):
        group_id = int(site.properties[GROUP_PROPERTY])
        indices_by_group.setdefault(group_id, []).append(index)
    for group_id, indices in indices_by_group.items():
        counts = plan.allocations[group_id]
        num_positions = len(indices)
        fractional = {
            species: count / num_positions
            for species, count in counts.items()
            if species != VACANCY and count > 0
        }
        if not fractional:
            raise ValueError(
                f"Occupancy group {group_id} has no occupied species to enumerate"
            )
        composition = Composition(fractional)
        for index in indices:
            supercell.replace(index, composition)
    for property_name in (
        GROUP_PROPERTY,
        CLUSTER_INSTANCE_PROPERTY,
        CLUSTER_ROLE_PROPERTY,
        CLUSTER_OCC_PROPERTY,
    ):
        if property_name in supercell.site_properties:
            supercell.remove_site_property(property_name)
    return supercell


def enumerate_ordered_candidates(
    decorated_structure: Structure,
    plan: AllocationPlan,
    groups: Sequence[OccupancyGroup],
    num_candidates: int,
    method: str,
    symprec: float,
    ewald_algo: int = 0,
) -> list[Structure] | None:
    """Exhaustively enumerate distinct orderings of the planned integer occupancies.

    Returns ``None`` — signalling the caller to fall back to random sampling — when
    the plan contains split-site pairs or clusters (whose local mutual exclusivity
    cannot be expressed as independent site occupancies), or when the requested
    backend is unavailable at runtime.
    """
    if plan.split_site_group_ids or plan.cluster_group_ids:
        return None

    supercell = plan_occupancy_supercell(decorated_structure, plan)
    if supercell.is_ordered:
        return [finalize_candidate_cell(supercell)]

    try:
        working = supercell.copy()
        try:
            working.add_oxidation_state_by_guess()
        except Exception:
            working = supercell
        if method == "enumerate":
            from pymatgen.transformations.advanced_transformations import (
                EnumerateStructureTransformation,
            )

            transformation = EnumerateStructureTransformation(
                min_cell_size=1,
                max_cell_size=1,
                symm_prec=symprec,
                sort_criteria="ewald",
            )
        elif method == "ewald":
            from pymatgen.transformations.standard_transformations import (
                OrderDisorderedStructureTransformation,
            )

            transformation = OrderDisorderedStructureTransformation(algo=ewald_algo)
        else:
            return None
        results = transformation.apply_transformation(
            working, return_ranked_list=max(1, num_candidates)
        )
    except Exception:
        return None

    if not isinstance(results, list):
        results = [{"structure": results}]

    matcher = StructureMatcher(
        primitive_cell=True, scale=False, attempt_supercell=False
    )
    candidates: list[Structure] = []
    for entry in results:
        structure = entry["structure"] if isinstance(entry, dict) else entry
        structure = structure.copy()
        structure.remove_oxidation_states()
        candidate = finalize_candidate_cell(structure)
        if not candidate.is_ordered:
            continue
        if any(matcher.fit(candidate, existing) for existing in candidates):
            continue
        candidates.append(candidate)
        if len(candidates) >= num_candidates:
            break
    return candidates or None


def generate_unique_candidates(
    decorated_structure: Structure,
    plan: AllocationPlan,
    groups: Sequence[OccupancyGroup],
    num_candidates: int,
    max_attempts_per_candidate: int,
    seed: int,
) -> list[Structure]:
    rng = random.Random(seed)
    matcher = StructureMatcher(
        primitive_cell=True,
        scale=False,
        attempt_supercell=False,
    )
    candidates: list[Structure] = []
    max_attempts = max(1, num_candidates * max_attempts_per_candidate)
    for _ in range(max_attempts):
        candidate = make_ordered_candidate(decorated_structure, plan, groups, rng)
        if any(matcher.fit(candidate, existing) for existing in candidates):
            continue
        candidates.append(candidate)
        if len(candidates) >= num_candidates:
            break
    return candidates


class Ranker:
    name = "none"

    def rank(
        self, structures: Sequence[Structure], label: str = ""
    ) -> list[RankedStructure]:
        return [
            RankedStructure(
                structure=structure,
                relaxed=False,
                total_energy_ev=None,
                energy_per_atom_ev=None,
            )
            for structure in structures
        ]


class CHGNetRanker(Ranker):
    name = "chgnet"

    def __init__(
        self, device: str, fmax: float, steps: int, relax_cell: bool
    ) -> None:
        try:
            from chgnet.model.dynamics import StructOptimizer
            from chgnet.model.model import CHGNet
        except ImportError as exc:
            raise RuntimeError(
                "CHGNet is not installed in the active environment. Install it in the "
                "workspace virtual environment or use --ranker mattersim/none."
            ) from exc
        self.optimizer = StructOptimizer(model=CHGNet.load(), use_device=device)
        self.fmax = fmax
        self.steps = steps
        self.relax_cell = relax_cell

    def rank(
        self, structures: Sequence[Structure], label: str = ""
    ) -> list[RankedStructure]:
        ranked: list[RankedStructure] = []
        for structure in structures:
            relaxation_start = time.monotonic()
            try:
                result = self.optimizer.relax(
                    structure,
                    fmax=self.fmax,
                    steps=self.steps,
                    relax_cell=self.relax_cell,
                    verbose=False,
                )
                relaxed = result["final_structure"]
                total_energy = float(result["trajectory"].energies[-1])
                relaxation_steps = max(0, len(result["trajectory"].energies) - 1)
                ranked.append(
                    RankedStructure(
                        structure=relaxed,
                        relaxed=True,
                        total_energy_ev=total_energy,
                        energy_per_atom_ev=total_energy / len(relaxed),
                        relaxation_steps=relaxation_steps,
                        relaxation_seconds=time.monotonic() - relaxation_start,
                    )
                )
            except Exception as exc:
                ranked.append(
                    RankedStructure(
                        structure=structure,
                        relaxed=False,
                        total_energy_ev=None,
                        energy_per_atom_ev=None,
                        error=f"{type(exc).__name__}: {exc}",
                        converged=False,
                        relaxation_seconds=time.monotonic() - relaxation_start,
                    )
                )
        return ranked


class MatterSimRanker(Ranker):
    name = "mattersim"

    def __init__(
        self,
        device: str,
        fmax: float,
        relax_cell: bool,
        steps: int,
        timeout: float,
        step_progress_every: int,
        max_natoms_per_batch: int,
        potential_path: Path | None,
    ) -> None:
        from mattersim.applications.batch_relax import BatchRelaxer
        from mattersim.forcefield.potential import Potential

        load_path = str(potential_path) if potential_path is not None else None
        potential = Potential.from_checkpoint(
            device=device,
            load_path=load_path,
            load_training_state=False,
        )

        class StepLimitedBatchRelaxer(BatchRelaxer):
            """Bound MatterSim relaxation and expose per-candidate step counts."""

            def __init__(self, *relaxer_args: Any, **relaxer_kwargs: Any) -> None:
                super().__init__(*relaxer_args, **relaxer_kwargs)
                self.max_steps = steps
                self.max_seconds = timeout
                self.progress_every = step_progress_every
                self.expected_count = 0
                self.batch_steps = 0
                self.label = ""
                self.step_counts: dict[int, int] = {}
                self.start_times: dict[int, float] = {}
                self.elapsed_seconds: dict[int, float] = {}
                self.converged_indices: set[int] = set()
                self.capped_indices: set[int] = set()
                self.timed_out_indices: set[int] = set()

            def insert(self, atoms: Any) -> None:
                super().insert(atoms)
                index = int(atoms.info["structure_index"])
                self.step_counts.setdefault(index, 0)
                self.start_times.setdefault(index, time.monotonic())

            def step_batch(self) -> None:
                active_before = {
                    int(opt.atoms.info["structure_index"])
                    for opt in self.optimizer_instances
                }
                super().step_batch()
                self.batch_steps += 1
                for index in active_before:
                    self.step_counts[index] += 1

                active_after = {
                    int(opt.atoms.info["structure_index"])
                    for opt in self.optimizer_instances
                }
                newly_converged = active_before - active_after
                self.converged_indices.update(newly_converged)

                retained_optimizers = []
                now = time.monotonic()
                for index in newly_converged:
                    self.elapsed_seconds[index] = now - self.start_times[index]
                for optimizer in self.optimizer_instances:
                    index = int(optimizer.atoms.info["structure_index"])
                    timed_out = (
                        self.max_seconds > 0
                        and now - self.start_times[index] >= self.max_seconds
                    )
                    if timed_out:
                        self.timed_out_indices.add(index)
                    if self.step_counts[index] >= self.max_steps or timed_out:
                        self.capped_indices.add(index)
                        self.elapsed_seconds[index] = now - self.start_times[index]
                    else:
                        retained_optimizers.append(optimizer)
                self.optimizer_instances = retained_optimizers
                self.is_active_instance = [True] * len(retained_optimizers)
                self.finished = not self.optimizer_instances

                if self.batch_steps % self.progress_every == 0:
                    self.print_step_summary(final=False)

            def print_step_summary(self, final: bool) -> None:
                values = list(self.step_counts.values())
                if not values:
                    return
                now = time.monotonic()
                elapsed_values = [
                    self.elapsed_seconds.get(index, now - self.start_times[index])
                    for index in self.step_counts
                ]
                prefix = "summary" if final else "progress"
                print(
                    f"MatterSim steps [{self.label}] {prefix}: "
                    f"inserted={len(values)}/{self.expected_count} "
                    f"active={len(self.optimizer_instances)} "
                    f"converged={len(self.converged_indices)} "
                    f"capped={len(self.capped_indices)} "
                    f"timed_out={len(self.timed_out_indices)} "
                    f"min/median/p90/max={min(values)}/"
                    f"{np.median(values):.1f}/{np.percentile(values, 90):.1f}/"
                    f"{max(values)} "
                    f"seconds[min/median/p90/max]={min(elapsed_values):.1f}/"
                    f"{np.median(elapsed_values):.1f}/"
                    f"{np.percentile(elapsed_values, 90):.1f}/"
                    f"{max(elapsed_values):.1f}",
                    flush=True,
                )

            def relax(
                self, atoms_list: list[Any], label: str = ""
            ) -> dict[int, list[Any]]:
                self.optimizer_instances = []
                self.is_active_instance = []
                self.finished = False
                self.expected_count = len(atoms_list)
                self.batch_steps = 0
                self.label = label
                self.step_counts = {}
                self.start_times = {}
                self.elapsed_seconds = {}
                self.converged_indices = set()
                self.capped_indices = set()
                self.timed_out_indices = set()
                trajectories = super().relax(atoms_list)
                self.print_step_summary(final=True)
                return trajectories

        self.relaxer = StepLimitedBatchRelaxer(
            potential=potential,
            filter="EXPCELLFILTER" if relax_cell else None,
            fmax=fmax,
            max_natoms_per_batch=max_natoms_per_batch,
        )

    def rank(
        self, structures: Sequence[Structure], label: str = ""
    ) -> list[RankedStructure]:
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
        try:
            trajectories = self.relaxer.relax(atoms, label=label)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return [
                RankedStructure(
                    structure=structure,
                    relaxed=False,
                    total_energy_ev=None,
                    energy_per_atom_ev=None,
                    error=message,
                    converged=False,
                    relaxation_steps=self.relaxer.step_counts.get(index),
                    relaxation_seconds=(
                        time.monotonic() - self.relaxer.start_times[index]
                        if index in self.relaxer.start_times
                        else None
                    ),
                )
                for index, structure in enumerate(structures)
            ]

        ranked: list[RankedStructure] = []
        for index, structure in enumerate(structures):
            try:
                final_atoms = trajectories[index][-1]
                relaxed = AseAtomsAdaptor.get_structure(final_atoms)
                total_energy = float(final_atoms.info["total_energy"])
                converged = index in self.relaxer.converged_indices
                if index in self.relaxer.timed_out_indices:
                    error = (
                        "Did not converge within "
                        f"{self.relaxer.max_seconds:g} seconds"
                    )
                elif index in self.relaxer.capped_indices:
                    error = f"Did not converge within {self.relaxer.max_steps} steps"
                else:
                    error = ""
                ranked.append(
                    RankedStructure(
                        structure=relaxed,
                        relaxed=True,
                        total_energy_ev=total_energy,
                        energy_per_atom_ev=total_energy / len(relaxed),
                        error=error,
                        converged=converged,
                        relaxation_steps=self.relaxer.step_counts.get(index),
                        relaxation_seconds=self.relaxer.elapsed_seconds.get(index),
                    )
                )
            except Exception as exc:
                ranked.append(
                    RankedStructure(
                        structure=structure,
                        relaxed=False,
                        total_energy_ev=None,
                        energy_per_atom_ev=None,
                        error=f"{type(exc).__name__}: {exc}",
                        converged=False,
                        relaxation_steps=self.relaxer.step_counts.get(index),
                        relaxation_seconds=self.relaxer.elapsed_seconds.get(index),
                    )
                )
        return ranked


def make_ranker(args: argparse.Namespace) -> Ranker:
    if args.ranker == "chgnet":
        return CHGNetRanker(
            device=args.device,
            fmax=args.fmax,
            steps=args.relax_steps,
            relax_cell=args.relax_cell,
        )
    if args.ranker == "mattersim":
        return MatterSimRanker(
            device=args.device,
            fmax=args.fmax,
            relax_cell=args.relax_cell,
            steps=args.relax_steps,
            timeout=args.relax_timeout,
            step_progress_every=args.step_progress_every,
            max_natoms_per_batch=args.max_natoms_per_batch,
            potential_path=args.potential_path,
        )
    return Ranker()


def parse_structure(path: Path) -> Structure:
    structures = CifParser(path).parse_structures(primitive=True, on_error="ignore")
    if not structures:
        raise ValueError("No structure parsed from CIF")
    return structures[0]


def energy_sort_key(
    item: tuple[int, RankedStructure],
) -> tuple[bool, bool, float, int]:
    index, ranked = item
    energy = ranked.energy_per_atom_ev
    return (
        ranked.converged is False,
        energy is None,
        float("inf") if energy is None else energy,
        index,
    )


def relaxation_outcome(ranked: RankedStructure) -> str:
    if ranked.converged is True:
        return "converged"
    if "seconds" in ranked.error:
        return "timeout"
    if "steps" in ranked.error:
        return "step_limit"
    if ranked.error:
        return "error"
    if ranked.converged is False:
        return "not_converged"
    return "not_evaluated"


def ordered_output_path(output_dir: Path, material_id: str, output_index: int) -> Path:
    suffix = "" if output_index == 0 else f"_{output_index:02d}"
    return output_dir / f"{material_id}_ordered{suffix}.cif"


def check_output_collisions(paths: Iterable[Path], overwrite: bool) -> None:
    if overwrite:
        return
    collisions = [path for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            "Output file already exists: "
            f"{collisions[0]}. Pass --overwrite to replace colliding files."
        )


def process_disordered_structure(
    path: Path,
    args: argparse.Namespace,
    ranker: Ranker,
    reference_composition: Composition | None,
    attempt: int = 0,
) -> tuple[list[CandidateResult], list[AnomalyResult]]:
    material_id = path.stem
    anomalies: list[AnomalyResult] = []
    source = parse_structure(path)
    source_formula = source.composition.formula
    reference_formula = (
        reference_composition.formula if reference_composition is not None else ""
    )
    source_composition_max_error: float | None = None
    reference_elements: set[str] = set()
    source_elements = {element.symbol for element in source.composition.elements}
    missing_reference_elements: list[str] = []
    if reference_composition is not None:
        reference_elements = {element.symbol for element in reference_composition.elements}
        missing_reference_elements = sorted(reference_elements - source_elements)
        source_composition_max_error = max(
            composition_errors(reference_composition, source.composition).values(),
            default=0.0,
        )

    if source.is_ordered:
        message = ""
        if missing_reference_elements:
            message = (
                "Reference elements absent from source CIF: "
                f"{', '.join(missing_reference_elements)}"
            )
        result = CandidateResult(
            parent_id=material_id,
            source_file=str(path),
            status="copied_ordered" if args.include_ordered else "skipped_ordered",
            attempt=attempt,
            message=message,
            reference_formula=reference_formula,
            source_formula=source_formula,
            cleaned_formula=source_formula,
            source_num_sites=len(source),
            source_composition_max_error=source_composition_max_error,
            composition_max_error=source_composition_max_error,
            composition_max_drift=0.0 if reference_composition is not None else None,
            within_composition_tolerance=(
                True if reference_composition is not None else None
            ),
            missing_reference_elements=",".join(missing_reference_elements),
            ranker=ranker.name,
        )
        if args.include_ordered:
            output_path = ordered_output_path(args.output_dir, material_id, 0)
            check_output_collisions([output_path], args.overwrite or args.resume)
            CifWriter(source).write_file(output_path)
            result.num_atoms = len(source)
            result.ordered_formula = source.composition.formula
            result.output_file = str(output_path)
        if message:
            anomalies.append(
                AnomalyResult(
                    parent_id=material_id,
                    source_file=str(path),
                    stage="input_validation",
                    status="reference_mismatch",
                    attempt=attempt,
                    message=message,
                    num_atoms=len(source),
                )
            )
        return [result], anomalies

    cleaned, dropped_count, dropped_sum = clean_small_occupancies(
        source,
        min_occupancy=args.min_occupancy,
        protected_elements=reference_elements,
    )

    split_site_clusters = (
        find_split_site_clusters(
            cleaned,
            max_distance=args.split_cluster_max_distance,
            occupancy_tolerance=args.split_cluster_occupancy_tolerance,
            symprec=args.symprec,
        )
        if args.detect_split_clusters
        else []
    )
    cluster_indices = {
        index
        for cluster in split_site_clusters
        for index in cluster.site_indices
    }
    split_site_pairs = (
        find_split_site_pairs(
            cleaned,
            max_distance=args.split_site_max_distance,
            occupancy_tolerance=args.split_site_occupancy_tolerance,
            excluded_indices=cluster_indices,
        )
        if args.detect_split_sites
        else []
    )
    groups, grouping_method = find_occupancy_groups(
        cleaned,
        symprec=args.symprec,
        split_site_pairs=split_site_pairs,
        split_site_clusters=split_site_clusters,
    )
    plan = build_allocation_plan(
        cleaned,
        groups,
        max_multiplier=args.max_supercell_multiplier,
        max_atoms=args.max_atoms,
        max_occupancy_error=args.max_occupancy_error,
        preserve_retained_species=args.preserve_retained_species,
        reference_composition=reference_composition,
        baseline_composition=source.composition,
        composition_tolerance=args.composition_tolerance,
        required_species=reference_elements & source_elements,
    )
    candidate_method = "random"
    candidates: list[Structure] | None = None
    if args.candidate_method != "random":
        ewald_algo = {"fast": 0, "complete": 1, "best_first": 2}[args.ewald_algo]
        backends = (
            ["enumerate", "ewald"]
            if args.candidate_method == "auto"
            else [args.candidate_method]
        )
        for backend in backends:
            candidates = enumerate_ordered_candidates(
                cleaned,
                plan,
                groups,
                num_candidates=args.num_candidates,
                method=backend,
                symprec=args.symprec,
                ewald_algo=ewald_algo,
            )
            if candidates:
                candidate_method = backend
                break
    if not candidates:
        candidates = generate_unique_candidates(
            cleaned,
            plan,
            groups,
            num_candidates=args.num_candidates,
            max_attempts_per_candidate=args.max_attempts_per_candidate,
            seed=stable_seed(args.seed + attempt * 1_000_003, material_id),
        )
        candidate_method = "random"
    if not candidates:
        raise RuntimeError("No unique ordered candidate was generated")

    ranked_candidates = ranker.rank(candidates, label=material_id)
    valid_ranked = [
        item
        for item in enumerate(ranked_candidates)
        if item[1].converged is not False
        and item[1].energy_per_atom_ev is not None
    ]
    has_valid_candidate = bool(valid_ranked)
    for candidate_index, ranked in enumerate(ranked_candidates):
        outcome = relaxation_outcome(ranked)
        if ranker.name != "none":
            energy = (
                "none"
                if ranked.energy_per_atom_ev is None
                else f"{ranked.energy_per_atom_ev:.8f}"
            )
            seconds = (
                "none"
                if ranked.relaxation_seconds is None
                else f"{ranked.relaxation_seconds:.1f}"
            )
            print(
                f"Candidate [{material_id} attempt={attempt} "
                f"{candidate_index + 1}/{len(ranked_candidates)}]: "
                f"status={outcome} steps={ranked.relaxation_steps} "
                f"seconds={seconds} energy_per_atom_ev={energy}"
                + (f" error={ranked.error}" if ranked.error else ""),
                flush=True,
            )
        if not ranked.error and ranked.converged is not False:
            continue
        if outcome == "timeout":
            anomaly_status = "relaxation_timeout"
        elif outcome == "step_limit":
            anomaly_status = "relaxation_step_limit"
        else:
            anomaly_status = "relaxation_error"
        anomalies.append(
            AnomalyResult(
                parent_id=material_id,
                source_file=str(path),
                stage="mlip_relaxation",
                status=anomaly_status,
                attempt=attempt,
                resolved=has_valid_candidate,
                resolution=(
                    "Excluded from ranking; a converged candidate was selected"
                    if has_valid_candidate
                    else ""
                ),
                message=ranked.error or "Relaxation did not converge",
                candidate_index=candidate_index,
                num_atoms=len(ranked.structure),
                converged=ranked.converged,
                relaxation_steps=ranked.relaxation_steps,
                relaxation_seconds=ranked.relaxation_seconds,
                total_energy_ev=ranked.total_energy_ev,
                energy_per_atom_ev=ranked.energy_per_atom_ev,
            )
        )
    indexed_ranked = sorted(
        valid_ranked or list(enumerate(ranked_candidates)), key=energy_sort_key
    )
    if args.keep_top > 0:
        indexed_ranked = indexed_ranked[: args.keep_top]

    output_paths = [
        ordered_output_path(args.output_dir, material_id, output_index)
        for output_index in range(len(indexed_ranked))
    ]
    check_output_collisions(output_paths, args.overwrite or args.resume)
    results: list[CandidateResult] = []
    for output_index, (candidate_index, ranked) in enumerate(indexed_ranked):
        energy_rank = output_index + 1
        output_path = output_paths[output_index]
        CifWriter(ranked.structure).write_file(output_path)
        messages = [ranked.error] if ranked.error else []
        if missing_reference_elements:
            messages.append(
                "Reference elements absent from source CIF: "
                f"{', '.join(missing_reference_elements)}"
            )
        if not plan.within_error_tolerance:
            messages.append(
                "Occupancy error "
                f"{plan.max_occupancy_error:.6f} exceeds tolerance "
                f"{args.max_occupancy_error:.6f}"
            )
        if plan.within_composition_tolerance is False:
            messages.append(
                "Added composition error "
                f"{plan.composition_max_drift:.6f} exceeds tolerance "
                f"{args.composition_tolerance:.6f}"
            )
        if ranked.error:
            status = "ranking_error"
        elif (
            plan.within_error_tolerance
            and plan.within_composition_tolerance is not False
            and not missing_reference_elements
        ):
            status = "ok"
        else:
            status = "approximation_warning"
        result = CandidateResult(
            parent_id=material_id,
            source_file=str(path),
            status=status,
            attempt=attempt,
            message="; ".join(messages),
            reference_formula=reference_formula,
            source_formula=source_formula,
            cleaned_formula=cleaned.composition.formula,
            source_num_sites=len(source),
            dropped_species_count=dropped_count,
            dropped_occupancy_sum=dropped_sum,
            grouping_method=grouping_method,
            candidate_method=candidate_method,
            split_site_pair_count=len(split_site_pairs),
            split_site_cluster_count=len(split_site_clusters),
            split_site_species=",".join(
                sorted(
                    {pair.species for pair in split_site_pairs}
                    | {cluster.species for cluster in split_site_clusters}
                )
            ),
            supercell_multiplier=plan.multiplier,
            supercell_matrix="x".join(str(value) for value in plan.scaling_matrix),
            occupancy_max_error=plan.max_occupancy_error,
            occupancy_mean_error=plan.mean_occupancy_error,
            within_error_tolerance=plan.within_error_tolerance,
            source_composition_max_error=source_composition_max_error,
            composition_max_error=plan.composition_max_error,
            composition_max_drift=plan.composition_max_drift,
            within_composition_tolerance=plan.within_composition_tolerance,
            missing_reference_elements=",".join(missing_reference_elements),
            candidate_index=candidate_index,
            energy_rank=energy_rank if ranker.name != "none" else None,
            num_atoms=len(ranked.structure),
            ordered_formula=ranked.structure.composition.formula,
            ranker=ranker.name,
            relaxed=ranked.relaxed,
            converged=ranked.converged,
            relaxation_steps=ranked.relaxation_steps,
            relaxation_seconds=ranked.relaxation_seconds,
            total_energy_ev=ranked.total_energy_ev,
            energy_per_atom_ev=ranked.energy_per_atom_ev,
            output_file=str(output_path),
        )
        results.append(result)
        if status == "approximation_warning":
            anomalies.append(
                AnomalyResult(
                    parent_id=material_id,
                    source_file=str(path),
                    stage="ordering_approximation",
                    status=status,
                    attempt=attempt,
                    message=result.message,
                    candidate_index=candidate_index,
                    num_atoms=result.num_atoms,
                    converged=result.converged,
                    relaxation_steps=result.relaxation_steps,
                    relaxation_seconds=result.relaxation_seconds,
                    total_energy_ev=result.total_energy_ev,
                    energy_per_atom_ev=result.energy_per_atom_ev,
                )
            )
    converged_count = sum(
        ranked.converged is True for ranked in ranked_candidates
    )
    anomalous_count = sum(
        ranked.converged is False for ranked in ranked_candidates
    )
    unevaluated_count = len(ranked_candidates) - converged_count - anomalous_count
    selected_indices = ",".join(
        str(candidate_index) for candidate_index, _ in indexed_ranked
    )
    print(
        f"Material [{material_id} attempt={attempt}] summary: "
        f"generated={len(ranked_candidates)} converged={converged_count} "
        f"anomalous={anomalous_count} unevaluated={unevaluated_count} "
        f"selected_candidate_indices={selected_indices or 'none'}",
        flush=True,
    )
    return results, anomalies


def select_input_files(args: argparse.Namespace) -> list[Path]:
    files = sorted(args.input_dir.glob("*.cif"))
    if args.ids:
        selected_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
        files = [path for path in files if path.stem in selected_ids]
        missing = sorted(selected_ids - {path.stem for path in files})
        if missing:
            raise FileNotFoundError(f"Unknown material IDs: {', '.join(missing)}")
    if args.start_after:
        positions = [
            index for index, path in enumerate(files) if path.stem == args.start_after
        ]
        if not positions:
            raise FileNotFoundError(
                f"Unknown --start-after material ID: {args.start_after}"
            )
        files = files[positions[0] + 1 :]
    if args.limit is not None:
        files = files[: args.limit]
    return files


def prepare_outputs(args: argparse.Namespace) -> None:
    resolved_output = args.output_dir.resolve()
    dangerous_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        args.input_dir.resolve(),
        args.input_dir.resolve().parent,
    }
    if resolved_output in dangerous_paths:
        raise ValueError(f"Refusing unsafe output directory: {resolved_output}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(f"Output directory is not a directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.anomaly_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        return
    for path in (args.report_path, args.anomaly_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Report already exists: {path}. Pass --overwrite to replace it."
            )
        if path.exists():
            path.unlink()


def load_records(path: Path, record_type: type[Any]) -> list[Any]:
    if not path.exists():
        return []
    field_names = {field.name for field in fields(record_type)}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [
            record_type(
                **{
                    name: value
                    for name, value in row.items()
                    if name in field_names and value is not None
                }
            )
            for row in reader
        ]


def write_records(path: Path, records: Iterable[Any], record_type: type[Any]) -> None:
    rows = [asdict(record) for record in records]
    fieldnames = [field.name for field in fields(record_type)]
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_report(path: Path, results: Iterable[CandidateResult]) -> None:
    write_records(path, results, CandidateResult)


def write_anomaly_report(path: Path, anomalies: Iterable[AnomalyResult]) -> None:
    write_records(path, anomalies, AnomalyResult)


def anomaly_from_existing_result(result: CandidateResult) -> AnomalyResult | None:
    stage_by_status = {
        "error": "processing",
        "ranking_error": "mlip_relaxation",
        "approximation_warning": "ordering_approximation",
    }
    stage = stage_by_status.get(result.status)
    if stage is None and result.message:
        stage = "input_validation"
    if stage is None:
        return None
    return AnomalyResult(
        parent_id=result.parent_id,
        source_file=result.source_file,
        stage=stage,
        status=result.status,
        attempt=int(result.attempt or 0),
        message=result.message,
        candidate_index=result.candidate_index,
        num_atoms=result.num_atoms,
        converged=result.converged,
        relaxation_steps=result.relaxation_steps,
        relaxation_seconds=result.relaxation_seconds,
        total_energy_ev=result.total_energy_ev,
        energy_per_atom_ev=result.energy_per_atom_ev,
    )


RETRYABLE_ANOMALY_STATUSES = {
    "error",
    "ranking_error",
    "relaxation_error",
    "relaxation_step_limit",
    "relaxation_timeout",
}


def csv_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def retryable_parent_ids(anomalies: Iterable[AnomalyResult]) -> set[str]:
    return {
        anomaly.parent_id
        for anomaly in anomalies
        if anomaly.status in RETRYABLE_ANOMALY_STATUSES
        and not csv_bool(anomaly.resolved)
    }


def next_attempts(
    results: Iterable[CandidateResult], parent_ids: Collection[str]
) -> dict[str, int]:
    attempts = {parent_id: 1 for parent_id in parent_ids}
    for result in results:
        if result.parent_id not in attempts:
            continue
        attempts[result.parent_id] = max(
            attempts[result.parent_id], int(result.attempt or 0) + 1
        )
    return attempts


def print_progress(
    index: int,
    total: int,
    start_time: float,
    path: Path,
    initial_completed: int = 0,
) -> None:
    elapsed = time.monotonic() - start_time
    completed_this_run = index - initial_completed
    rate = completed_this_run / elapsed if elapsed > 0 else 0.0
    remaining = (total - index) / rate if rate > 0 else float("inf")
    eta = "unknown" if not math.isfinite(remaining) else f"{remaining / 60:.1f} min"
    print(
        f"[{index}/{total}] {path.name} | elapsed={elapsed / 60:.1f} min | ETA={eta}",
        flush=True,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if args.report_path.resolve() == args.anomaly_path.resolve():
        raise ValueError("--report-path and --anomaly-path must be different")
    if not 0 <= args.min_occupancy < 1:
        raise ValueError("--min-occupancy must be in [0, 1)")
    if args.split_site_max_distance <= 0:
        raise ValueError("--split-site-max-distance must be positive")
    if args.split_site_occupancy_tolerance < 0:
        raise ValueError("--split-site-occupancy-tolerance cannot be negative")
    if args.split_cluster_max_distance <= 0:
        raise ValueError("--split-cluster-max-distance must be positive")
    if args.split_cluster_occupancy_tolerance < 0:
        raise ValueError("--split-cluster-occupancy-tolerance cannot be negative")
    if args.max_supercell_multiplier < 1:
        raise ValueError("--max-supercell-multiplier must be positive")
    if args.max_atoms < 1 or args.num_candidates < 1:
        raise ValueError("--max-atoms and --num-candidates must be positive")
    if args.max_occupancy_error < 0:
        raise ValueError("--max-occupancy-error cannot be negative")
    if args.composition_tolerance < 0:
        raise ValueError("--composition-tolerance cannot be negative")
    if args.keep_top < 0:
        raise ValueError("--keep-top cannot be negative")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.relax_steps < 1:
        raise ValueError("--relax-steps must be positive")
    if args.relax_timeout < 0:
        raise ValueError("--relax-timeout cannot be negative")
    if args.step_progress_every < 1:
        raise ValueError("--step-progress-every must be positive")


def main() -> int:
    args = build_parser().parse_args()
    if args.retry_anomalies:
        args.resume = True
    if args.anomaly_path is None:
        suffix = args.report_path.suffix or ".csv"
        args.anomaly_path = args.report_path.with_name(
            f"{args.report_path.stem}_anomalies{suffix}"
        )
    validate_args(args)
    selected_files = select_input_files(args)
    if not selected_files:
        raise FileNotFoundError(f"No CIF files found under {args.input_dir}")
    reference_compositions = load_reference_compositions(args.composition_csv)
    prepare_outputs(args)

    results = (
        load_records(args.report_path, CandidateResult) if args.resume else []
    )
    anomalies = (
        load_records(args.anomaly_path, AnomalyResult) if args.resume else []
    )
    if args.resume and not args.anomaly_path.exists():
        anomalies = [
            anomaly
            for result in results
            if (anomaly := anomaly_from_existing_result(result)) is not None
        ]
    attempt_by_id: dict[str, int] = {}
    if args.retry_anomalies:
        retry_ids = retryable_parent_ids(anomalies)
        if not retry_ids:
            print(
                f"Done. No unresolved retryable anomalies in {args.anomaly_path}",
                flush=True,
            )
            return 0
        selected_ids = {path.stem for path in selected_files}
        missing_retry_ids = sorted(retry_ids - selected_ids)
        if missing_retry_ids:
            raise FileNotFoundError(
                "Retry inputs are missing: " + ", ".join(missing_retry_ids)
            )
        attempt_by_id = next_attempts(results, retry_ids)
        retry_plan = ", ".join(
            f"{parent_id}(attempt={attempt_by_id[parent_id]})"
            for parent_id in sorted(retry_ids)
        )
        print(f"Retry plan: {retry_plan}", flush=True)
        results = [result for result in results if result.parent_id not in retry_ids]
        anomalies = [
            anomaly
            for anomaly in anomalies
            if anomaly.parent_id not in retry_ids
        ]
        selected_files = [
            path for path in selected_files if path.stem in retry_ids
        ]
    completed_ids = {result.parent_id for result in results}
    pending_files = [
        path for path in selected_files if path.stem not in completed_ids
    ]
    initial_completed = len(selected_files) - len(pending_files)

    print(
        f"Processing {len(selected_files)} CIFs with ranker={args.ranker}; "
        f"resumed={initial_completed}, pending={len(pending_files)}. "
        f"retry_anomalies={args.retry_anomalies}. "
        "MLIP ranking selects low-energy representatives; it does not recover a unique "
        "experimental ordering.",
        flush=True,
    )
    if not pending_files:
        print(
            f"Done. All selected materials are already present in {args.report_path}",
            flush=True,
        )
        return 0

    ranker = make_ranker(args)
    start_time = time.monotonic()
    for index, path in enumerate(pending_files, start=initial_completed + 1):
        attempt = attempt_by_id.get(path.stem, 0)
        caught_warnings: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                material_results, material_anomalies = process_disordered_structure(
                    path,
                    args,
                    ranker,
                    reference_compositions.get(path.stem),
                    attempt=attempt,
                )
            results.extend(material_results)
            anomalies.extend(material_anomalies)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            results.append(
                CandidateResult(
                    parent_id=path.stem,
                    source_file=str(path),
                    status="error",
                    attempt=attempt,
                    message=message,
                    ranker=ranker.name,
                )
            )
            anomalies.append(
                AnomalyResult(
                    parent_id=path.stem,
                    source_file=str(path),
                    stage="processing",
                    status="error",
                    attempt=attempt,
                    message=message,
                )
            )
        anomalies.extend(
            AnomalyResult(
                parent_id=path.stem,
                source_file=str(path),
                stage="python_warning",
                status=warning.category.__name__,
                attempt=attempt,
                message=f"{warning.filename}:{warning.lineno}: {warning.message}",
            )
            for warning in caught_warnings
        )
        write_report(args.report_path, results)
        write_anomaly_report(args.anomaly_path, anomalies)
        if (
            (index - initial_completed) % args.progress_every == 0
            or index == len(selected_files)
        ):
            print_progress(
                index,
                len(selected_files),
                start_time,
                path,
                initial_completed=initial_completed,
            )

    ok = sum(result.status in {"ok", "approximation_warning"} for result in results)
    warning_count = sum(
        result.status == "approximation_warning" for result in results
    )
    errors = sum(result.status in {"error", "ranking_error"} for result in results)
    unresolved_anomalies = retryable_parent_ids(anomalies)
    resolved_relaxation_anomalies = sum(
        anomaly.stage == "mlip_relaxation" and csv_bool(anomaly.resolved)
        for anomaly in anomalies
    )
    print(
        f"Done. report={args.report_path} anomalies={args.anomaly_path} candidates={ok} "
        f"warnings={warning_count} errors={errors} "
        f"resolved_candidate_anomalies={resolved_relaxation_anomalies} "
        f"unresolved_anomaly_materials={len(unresolved_anomalies)}",
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
