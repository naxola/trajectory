"""Tests para data/datasets.py"""
import torch
from surgical_fl.data.generators.factory import build_generator
from surgical_fl.data.datasets import TrajectoryDataset, build_dataloader


class TestTrajectoryDataset:

    def setup_method(self):
        gen = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        self.trajectories = gen.generate(20)
        self.ds = TrajectoryDataset(self.trajectories)

    def test_length(self):
        assert len(self.ds) == 20

    def test_item_shapes(self):
        inp, tgt = self.ds[0]
        assert inp.shape == (49, 2)   # T-1 porque predice next point
        assert tgt.shape == (49, 2)

    def test_item_dtype(self):
        inp, tgt = self.ds[0]
        assert inp.dtype == torch.float32
        assert tgt.dtype == torch.float32

    def test_target_is_shifted_input(self):
        """Target[t] debe coincidir con Input[t+1] (secuencia desplazada)."""
        inp, tgt = self.ds[0]
        # tgt[0] == punto 1 de la trayectoria original
        # inp[1] == punto 1 de la trayectoria original
        torch.testing.assert_close(tgt[0], inp[1])


class TestBuildDataloader:

    def test_returns_dataloader(self):
        gen = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        loader = build_dataloader(gen, n_samples=16, batch_size=8)
        batch_in, batch_tgt = next(iter(loader))
        assert batch_in.shape[0] <= 8
        assert batch_in.shape[1] == 49
        assert batch_in.shape[2] == 2
