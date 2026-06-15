"""Tests para data/generators/trajectories/cutting.py"""
import numpy as np
import pytest
from surgical_fl.data.generators.trajectories.cutting import (
    LinearCutGenerator,
    CurvedCutGenerator,
    SplineCutGenerator,
)


class TestLinearCutGenerator:

    def test_shape(self):
        gen = LinearCutGenerator(trajectory_length=50, seed=0)
        data = gen.generate(10)
        assert data.shape == (10, 50, 2)

    def test_dtype(self):
        gen = LinearCutGenerator(trajectory_length=50, seed=0)
        data = gen.generate(5)
        assert data.dtype == np.float32

    def test_x_monotonically_increasing(self):
        gen = LinearCutGenerator(trajectory_length=50, seed=0)
        data = gen.generate(10)
        for traj in data:
            assert np.all(np.diff(traj[:, 0]) > 0)

    def test_x_normalized_to_unit(self):
        gen = LinearCutGenerator(trajectory_length=50, seed=0)
        data = gen.generate(10)
        for traj in data:
            assert abs(traj[-1, 0] - 1.0) < 1e-5

    def test_generate_one(self):
        gen = LinearCutGenerator(trajectory_length=30, seed=0)
        single = gen.generate_one()
        assert single.shape == (30, 2)


class TestCurvedCutGenerator:

    def test_shape(self):
        gen = CurvedCutGenerator(trajectory_length=30, seed=0)
        data = gen.generate(5)
        assert data.shape == (5, 30, 2)

    def test_generate_one(self):
        gen = CurvedCutGenerator(trajectory_length=40, seed=0)
        single = gen.generate_one()
        assert single.shape == (40, 2)

    def test_curvature_visible_in_y(self):
        """Con curvatura alta y ruido cero, Y debe tener forma parabólica."""
        gen = CurvedCutGenerator(
            trajectory_length=50, curvature=0.5, noise_std=0.0, seed=0,
        )
        traj = gen.generate_one()
        # Los extremos deben tener Y mayor que el centro (parábola)
        y_edges = (traj[0, 1] + traj[-1, 1]) / 2
        y_center = traj[25, 1]
        assert y_edges > y_center


class TestSplineCutGenerator:

    def test_shape(self):
        gen = SplineCutGenerator(trajectory_length=50, seed=0)
        data = gen.generate(10)
        assert data.shape == (10, 50, 2)

    def test_dtype(self):
        gen = SplineCutGenerator(trajectory_length=50, seed=0)
        data = gen.generate(5)
        assert data.dtype == np.float32

    def test_generate_one(self):
        gen = SplineCutGenerator(trajectory_length=40, seed=0)
        single = gen.generate_one()
        assert single.shape == (40, 2)

    def test_endpoints_on_reference_without_perturbation(self):
        """Con radio 0 y ruido 0, el corte coincide con los extremos de la
        incisión de referencia por defecto: (0,0) y (1,0)."""
        gen = SplineCutGenerator(trajectory_length=50, radius=0.0, noise_std=0.0, seed=0)
        traj = gen.generate_one()
        np.testing.assert_allclose(traj[0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(traj[-1], [1.0, 0.0], atol=1e-6)

    def test_larger_radius_more_deviation(self):
        """Un radio mayor (mano menos precisa) produce más desviación en Y."""
        gen_small = SplineCutGenerator(trajectory_length=80, radius=0.02, noise_std=0.0, seed=0)
        gen_big = SplineCutGenerator(trajectory_length=80, radius=0.10, noise_std=0.0, seed=0)
        var_small = np.var(gen_small.generate(50)[:, :, 1])
        var_big = np.var(gen_big.generate(50)[:, :, 1])
        assert var_big > var_small

    def test_more_nodes_than_reference_allowed(self):
        """El corte simulado puede tener más nodos que la curva de referencia."""
        gen = SplineCutGenerator(trajectory_length=60, n_nodes=8, seed=0)
        assert gen.n_nodes > len(gen.reference_nodes)
        assert gen.generate(3).shape == (3, 60, 2)

    def test_reference_curve_shape_and_endpoints(self):
        """reference_curve es una curva densa de (0,0) a (1,0)."""
        gen = SplineCutGenerator(trajectory_length=50, seed=0)
        ref = gen.reference_curve
        assert ref.ndim == 2 and ref.shape[1] == 2
        np.testing.assert_allclose(ref[0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(ref[-1], [1.0, 0.0], atol=1e-6)

    def test_shared_reference_across_instances(self):
        """Dos hospitales spline con distinto radio comparten la misma
        referencia → son comparables."""
        gen_expert = SplineCutGenerator(radius=0.03, seed=1)
        gen_novice = SplineCutGenerator(radius=0.08, seed=2)
        np.testing.assert_array_equal(
            gen_expert.reference_curve, gen_novice.reference_curve
        )


class TestGeneratorMetadata:
    """Regresión: metadata/validate_output construyen SurgicalGeneratorMetadata.

    Antes fallaban con TypeError por un typo (output_shpae) en el dataclass.
    """

    def test_metadata_has_expected_output_shape(self):
        gen = LinearCutGenerator(trajectory_length=40, seed=0)
        meta = gen.metadata
        assert meta.skill == "cutting"
        assert meta.output_type == "trajectory"
        assert meta.output_shape == (40, 2)
        assert meta.units == "meters"

    def test_metadata_accessible_for_all_cutting_generators(self):
        for gen in (
            LinearCutGenerator(trajectory_length=30, seed=0),
            CurvedCutGenerator(trajectory_length=30, seed=0),
            SplineCutGenerator(trajectory_length=30, seed=0),
        ):
            assert gen.metadata.output_shape == (30, 2)

    def test_validate_output_accepts_correct_shape(self):
        gen = LinearCutGenerator(trajectory_length=50, seed=0)
        assert gen.validate_output(gen.generate(5)) is True

    def test_validate_output_rejects_wrong_shape(self):
        gen = LinearCutGenerator(trajectory_length=50, seed=0)
        bad = np.zeros((5, 50, 3), dtype=np.float32)  # 3D en vez de 2D
        with pytest.raises(ValueError, match="Shape inesperado"):
            gen.validate_output(bad)


class TestReproducibility:

    def test_same_seed_same_output(self):
        gen1 = LinearCutGenerator(trajectory_length=50, seed=42)
        gen2 = LinearCutGenerator(trajectory_length=50, seed=42)
        np.testing.assert_array_equal(gen1.generate(5), gen2.generate(5))

    def test_different_seed_different_output(self):
        gen1 = LinearCutGenerator(trajectory_length=50, seed=0)
        gen2 = LinearCutGenerator(trajectory_length=50, seed=99)
        assert not np.array_equal(gen1.generate(5), gen2.generate(5))
