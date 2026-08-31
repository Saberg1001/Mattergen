import itertools
import random
import tempfile
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

    def test_converged_candidates_sort_before_capped_candidates(self) -> None:
        source = ordering.parse_structure(DATA_DIR / "cifs" / "duf.cif")
        capped = ordering.RankedStructure(
            structure=source,
            relaxed=True,
            total_energy_ev=-100.0,
            energy_per_atom_ev=-10.0,
            error="Did not converge within 300 steps",
            converged=False,
            relaxation_steps=300,
        )
        converged = ordering.RankedStructure(
            structure=source,
            relaxed=True,
            total_energy_ev=-1.0,
            energy_per_atom_ev=-0.1,
            converged=True,
            relaxation_steps=25,
        )

        ranked = sorted(enumerate([capped, converged]), key=ordering.energy_sort_key)

        self.assertEqual(ranked[0][0], 1)

    def test_reports_round_trip_resume_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "report.csv"
            anomaly_path = Path(temporary_dir) / "anomalies.csv"
            result = ordering.CandidateResult(
                parent_id="sample",
                source_file="sample.cif",
                status="ok",
                converged=True,
                relaxation_steps=42,
            )
            anomaly = ordering.AnomalyResult(
                parent_id="sample",
                source_file="sample.cif",
                stage="mlip_relaxation",
                status="relaxation_timeout",
                relaxation_steps=42,
            )

            ordering.write_report(report_path, [result])
            ordering.write_anomaly_report(anomaly_path, [anomaly])
            loaded_results = ordering.load_records(
                report_path, ordering.CandidateResult
            )
            loaded_anomalies = ordering.load_records(
                anomaly_path, ordering.AnomalyResult
            )

            self.assertEqual(loaded_results[0].parent_id, "sample")
            self.assertEqual(loaded_results[0].relaxation_steps, "42")
            self.assertEqual(loaded_anomalies[0].status, "relaxation_timeout")

    def test_existing_warning_is_backfilled_as_anomaly(self) -> None:
        result = ordering.CandidateResult(
            parent_id="sample",
            source_file="sample.cif",
            status="approximation_warning",
            message="Occupancy error exceeds tolerance",
        )

        anomaly = ordering.anomaly_from_existing_result(result)

        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.stage, "ordering_approximation")
        self.assertEqual(anomaly.status, "approximation_warning")

    def test_retry_ignores_candidate_failure_resolved_by_other_candidate(self) -> None:
        resolved = ordering.AnomalyResult(
            parent_id="resolved",
            source_file="resolved.cif",
            stage="mlip_relaxation",
            status="relaxation_timeout",
            resolved=True,
        )
        unresolved = ordering.AnomalyResult(
            parent_id="unresolved",
            source_file="unresolved.cif",
            stage="mlip_relaxation",
            status="relaxation_step_limit",
        )

        retry_ids = ordering.retryable_parent_ids([resolved, unresolved])

        self.assertEqual(retry_ids, {"unresolved"})

    def test_retry_increments_attempt_number(self) -> None:
        results = [
            ordering.CandidateResult(
                parent_id="sample",
                source_file="sample.cif",
                status="ranking_error",
                attempt=2,
            )
        ]

        attempts = ordering.next_attempts(results, {"sample"})

        self.assertEqual(attempts["sample"], 3)

    def test_failed_candidate_is_excluded_when_another_converges(self) -> None:
        class MixedRanker(ordering.Ranker):
            name = "mattersim"

            def rank(self, structures, label=""):
                return [
                    ordering.RankedStructure(
                        structure=structures[0],
                        relaxed=True,
                        total_energy_ev=-1.0,
                        energy_per_atom_ev=-0.1,
                        converged=True,
                        relaxation_steps=20,
                    ),
                    ordering.RankedStructure(
                        structure=structures[1],
                        relaxed=True,
                        total_energy_ev=-100.0,
                        energy_per_atom_ev=-10.0,
                        error="Did not converge within 300 steps",
                        converged=False,
                        relaxation_steps=300,
                    ),
                ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "cifs"
            output_dir.mkdir()
            args = ordering.build_parser().parse_args(
                [
                    "--output-dir",
                    str(output_dir),
                    "--num-candidates",
                    "2",
                    "--keep-top",
                    "1",
                    "--overwrite",
                ]
            )
            references = ordering.load_reference_compositions(DATA_DIR / "all.csv")

            results, anomalies = ordering.process_disordered_structure(
                DATA_DIR / "cifs" / "019.cif",
                args,
                MixedRanker(),
                references["019"],
            )

            self.assertEqual(results[0].candidate_index, 0)
            self.assertEqual(results[0].status, "ok")
            self.assertEqual(len(anomalies), 1)
            self.assertTrue(anomalies[0].resolved)
            self.assertEqual(
                anomalies[0].status, "relaxation_step_limit"
            )

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
