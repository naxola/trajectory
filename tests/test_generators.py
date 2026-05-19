"""Tests para data/generators/trajectories/cutting.py"""
import numpy as np
from surgical_fl.data.generators.trajectories.cutting import (
    LinearCutGenerator,
    CurvedCutGenerator,
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


class TestReproducibility:

    def test_same_seed_same_output(self):
        gen1 = LinearCutGenerator(trajectory_length=50, seed=42)
        gen2 = LinearCutGenerator(trajectory_length=50, seed=42)
        np.testing.assert_array_equal(gen1.generate(5), gen2.generate(5))

    def test_different_seed_different_output(self):
        gen1 = LinearCutGenerator(trajectory_length=50, seed=0)
        gen2 = LinearCutGenerator(trajectory_length=50, seed=99)
        assert not np.array_equal(gen1.generate(5), gen2.generate(5))
