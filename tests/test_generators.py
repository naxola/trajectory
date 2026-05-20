"""Tests para data/generators/trajectories/cutting.py"""
import numpy as np
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

    def test_x_is_uniform_linspace(self):
        """X debe ser un linspace uniforme de 0 a 1."""
        gen = SplineCutGenerator(trajectory_length=50, seed=0)
        traj = gen.generate_one()
        expected_x = np.linspace(0, 1, 50)
        np.testing.assert_allclose(traj[:, 0], expected_x, atol=1e-6)

    def test_endpoints_zero_without_noise(self):
        """Con ruido cero, Y en los extremos debe ser exactamente 0 (puntos de control fijados a 0)."""
        gen = SplineCutGenerator(trajectory_length=50, noise_std=0.0, seed=0)
        traj = gen.generate_one()
        assert abs(traj[0, 1]) < 1e-6
        assert abs(traj[-1, 1]) < 1e-6

    def test_more_control_points_more_variation(self):
        """Más puntos de control permiten mayor variación en Y (con ruido cero)."""
        gen_few = SplineCutGenerator(trajectory_length=100, n_control_points=2, noise_std=0.0, seed=0)
        gen_many = SplineCutGenerator(trajectory_length=100, n_control_points=10, noise_std=0.0, seed=0)
        data_few = gen_few.generate(50)
        data_many = gen_many.generate(50)
        var_few = np.var(data_few[:, :, 1])
        var_many = np.var(data_many[:, :, 1])
        assert var_many > var_few


class TestReproducibility:

    def test_same_seed_same_output(self):
        gen1 = LinearCutGenerator(trajectory_length=50, seed=42)
        gen2 = LinearCutGenerator(trajectory_length=50, seed=42)
        np.testing.assert_array_equal(gen1.generate(5), gen2.generate(5))

    def test_different_seed_different_output(self):
        gen1 = LinearCutGenerator(trajectory_length=50, seed=0)
        gen2 = LinearCutGenerator(trajectory_length=50, seed=99)
        assert not np.array_equal(gen1.generate(5), gen2.generate(5))
