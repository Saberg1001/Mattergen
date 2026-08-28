import itertools
import random
import unittest
from dataclasses import replace
from pathlib import Path

from pymatgen.core import Composition

from OBELIX import order_disordered_cifs as ordering


DATA_DIR = Path(__file__).parent


class OrderingCompositionTest(unittest.TestCase):
    def test_2t8_split_sites_are_mutually_exclusive(self) -> None:
        references = ordering.load_reference_compositions(DATA_DIR / "all.csv")
        source = ordering.parse_structure(DATA_DIR / "cifs" / "2t8.cif")
        reference = references["2t8"]
        reference_elements = {element.symbol for element in reference.elements}
        source_elements = {element.symbol for element in source.composition.elements}
        cleaned, _, _ = ordering.clean_small_occupancies(
            source,
            min_occupancy=0.01,
            occupancy_sum_tolerance=1e-5,
            protected_elements=reference_elements,
        )

        pairs = ordering.find_split_site_pairs(
            cleaned,
            max_distance=1.2,
            occupancy_tolerance=0.05,
        )
        groups, _ = ordering.find_occupancy_groups(
            cleaned,
            symprec=0.1,
            occupancy_sum_tolerance=1e-5,
            split_site_pairs=pairs,
        )
        plan = ordering.build_allocation_plan(
            cleaned,
            groups,
            max_multiplier=8,
            max_atoms=128,
            max_occupancy_error=0.05,
            preserve_retained_species=True,
            reference_composition=reference,
            baseline_composition=source.composition,
            composition_tolerance=0.05,
            required_species=reference_elements & source_elements,
        )
        candidates = ordering.generate_unique_candidates(
            cleaned,
            plan,
            groups,
            num_candidates=5,
            max_attempts_per_candidate=20,
            seed=ordering.stable_seed(17, "2t8"),
        )
        doubled_allocations = {
            group_id: {species: count * 2 for species, count in counts.items()}
            for group_id, counts in plan.allocations.items()
        }
        doubled_plan = replace(
            plan,
            multiplier=2,
            scaling_matrix=(2, 1, 1),
            allocations=doubled_allocations,
            estimated_num_atoms=plan.estimated_num_atoms * 2,
        )
        candidates.append(
            ordering.make_ordered_candidate(
                cleaned,
                doubled_plan,
                groups,
                random.Random(23),
            )
        )

        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair.species == "Li" for pair in pairs))
        self.assertTrue(all(abs(pair.distance - 0.668239) < 1e-5 for pair in pairs))
        for candidate in candidates:
            lithium_indices = [
                index
                for index, site in enumerate(candidate)
                if site.specie.symbol == "Li"
            ]
            lithium_distances = [
                candidate.get_distance(first, second)
                for first, second in itertools.combinations(lithium_indices, 2)
            ]
            self.assertGreater(min(lithium_distances), 1.2)

    def test_09w_three_position_li_clusters_are_mutually_exclusive(self) -> None:
        references = ordering.load_reference_compositions(DATA_DIR / "all.csv")
        source = ordering.parse_structure(DATA_DIR / "cifs" / "09w.cif")
        reference = references["09w"]
        reference_elements = {element.symbol for element in reference.elements}
        source_elements = {element.symbol for element in source.composition.elements}
        cleaned, _, _ = ordering.clean_small_occupancies(
            source,
            min_occupancy=0.01,
            occupancy_sum_tolerance=1e-5,
            protected_elements=reference_elements,
        )

        clusters = ordering.find_split_site_clusters(
            cleaned,
            max_distance=1.5,
            occupancy_tolerance=0.1,
            symprec=0.1,
        )
        cluster_indices = {
            index for cluster in clusters for index in cluster.site_indices
        }
        pairs = ordering.find_split_site_pairs(
            cleaned,
            max_distance=1.2,
            occupancy_tolerance=0.1,
            excluded_indices=cluster_indices,
        )
        groups, _ = ordering.find_occupancy_groups(
            cleaned,
            symprec=0.1,
            occupancy_sum_tolerance=1e-5,
            split_site_pairs=pairs,
            split_site_clusters=clusters,
        )
        plan = ordering.build_allocation_plan(
            cleaned,
            groups,
            max_multiplier=8,
            max_atoms=128,
            max_occupancy_error=0.05,
            preserve_retained_species=True,
            reference_composition=reference,
            baseline_composition=source.composition,
            composition_tolerance=0.05,
            required_species=reference_elements & source_elements,
        )
        candidates = ordering.generate_unique_candidates(
            cleaned,
            plan,
            groups,
            num_candidates=3,
            max_attempts_per_candidate=20,
            seed=ordering.stable_seed(17, "09w"),
        )

        self.assertEqual(len(clusters), 6)
        self.assertEqual(len({cluster.family_id for cluster in clusters}), 1)
        self.assertTrue(all(cluster.cluster_size == 3 for cluster in clusters))
        self.assertTrue(
            all(abs(cluster.occupancy_sum - 1.00833) < 1e-5 for cluster in clusters)
        )
        self.assertEqual(len(pairs), 0)
        cluster_group_id = plan.cluster_group_ids[0]
        self.assertEqual(plan.cluster_center_counts[cluster_group_id], 6)
        for candidate in candidates:
            self.assertEqual(candidate.composition["Li"], 48)
            lithium_indices = [
                index
                for index, site in enumerate(candidate)
                if site.specie.symbol == "Li"
            ]
            lithium_distances = [
                candidate.get_distance(first, second)
                for first, second in itertools.combinations(lithium_indices, 2)
            ]
            self.assertGreater(min(lithium_distances), 1.5)

    def test_ordered_outputs_are_flat(self) -> None:
        output_dir = Path("ordered_cifs")

        self.assertEqual(
            ordering.ordered_output_path(output_dir, "2t8", 0),
            output_dir / "2t8_ordered.cif",
        )
        self.assertEqual(
            ordering.ordered_output_path(output_dir, "2t8", 1),
            output_dir / "2t8_ordered_01.cif",
        )

    def test_existing_csv_difference_is_baseline(self) -> None:
        reference = Composition("LiO")
        baseline = Composition("Li0.8O1.2")

        max_error, max_drift = ordering.composition_error_metrics(
            reference,
            baseline,
            baseline,
        )

        self.assertAlmostEqual(max_error, 0.1)
        self.assertAlmostEqual(max_drift, 0.0)

    def test_trace_reference_element_survives_ordering(self) -> None:
        references = ordering.load_reference_compositions(DATA_DIR / "all.csv")
        source = ordering.parse_structure(DATA_DIR / "cifs" / "duf.cif")
        reference = references["duf"]
        reference_elements = {element.symbol for element in reference.elements}
        source_elements = {element.symbol for element in source.composition.elements}

        cleaned, dropped_count, _ = ordering.clean_small_occupancies(
            source,
            min_occupancy=0.01,
            occupancy_sum_tolerance=1e-5,
            protected_elements=reference_elements,
        )
        groups, _ = ordering.find_occupancy_groups(
            cleaned,
            symprec=0.1,
            occupancy_sum_tolerance=1e-5,
        )
        plan = ordering.build_allocation_plan(
            cleaned,
            groups,
            max_multiplier=8,
            max_atoms=128,
            max_occupancy_error=0.05,
            preserve_retained_species=True,
            reference_composition=reference,
            baseline_composition=source.composition,
            composition_tolerance=0.05,
            required_species=reference_elements & source_elements,
        )

        self.assertEqual(dropped_count, 0)
        self.assertIn("Sb", cleaned.composition)
        self.assertGreaterEqual(ordering.allocation_counts(plan.allocations)["Sb"], 1)
        self.assertTrue(plan.within_composition_tolerance)


if __name__ == "__main__":
    unittest.main()
