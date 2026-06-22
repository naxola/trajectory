"""Tests para federation/ — FlowerClient, FedAvg, config de federación."""
import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from surgical_fl.data.builders import build_centralized_split
from surgical_fl.data.datasets import TrajectoryDataset
from surgical_fl.models.registry import get_model
from surgical_fl.federation.client import FlowerClient
from surgical_fl.training.trainer import evaluate
from surgical_fl.utils.config import ExperimentConfig, FederationConfig


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def split():
    return build_centralized_split(
        skill="cutting",
        profile_names=["hospital_c", "hospital_d"],
        total_samples=80,
        val_split=0.2,
        trajectory_length=20,
        seed=42,
    )


@pytest.fixture()
def model():
    return get_model("rnn", hidden_dim=16, dropout=0.0)


@pytest.fixture()
def device():
    return torch.device("cpu")


def _make_client(profile, split, model, device, local_epochs=1):
    train_ds = TrajectoryDataset(split.train_per_profile[profile])
    val_ds = split.val_per_profile[profile]
    return FlowerClient(
        model=model,
        train_loader=DataLoader(train_ds, batch_size=16, shuffle=True),
        val_loader=DataLoader(val_ds, batch_size=16, shuffle=False),
        local_epochs=local_epochs,
        learning_rate=0.001,
        device=device,
    )


# ── FlowerClient ────────────────────────────────────────────────────────────

class TestFlowerClient:

    def test_get_parameters_returns_ndarrays(self, split, model, device):
        client = _make_client("hospital_c", split, model, device)
        params = client.get_parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters_changes_weights(self, split, model, device):
        client = _make_client("hospital_c", split, model, device)
        original = client.get_parameters()
        zeros = [np.zeros_like(p) for p in original]
        client.set_parameters(zeros)
        for p in client.get_parameters():
            np.testing.assert_array_equal(p, np.zeros_like(p))

    def test_fit_returns_updated_params(self, split, model, device):
        client = _make_client("hospital_c", split, model, device)
        params_before = [p.copy() for p in client.get_parameters()]
        new_params, n_samples, metrics = client.fit(params_before, {})
        assert isinstance(new_params, list)
        assert n_samples > 0
        assert "train_loss" in metrics
        changed = any(
            not np.array_equal(a, b) for a, b in zip(params_before, new_params)
        )
        assert changed, "fit() debe modificar los pesos"

    def test_evaluate_returns_loss_and_metrics(self, split, model, device):
        client = _make_client("hospital_c", split, model, device)
        params = client.get_parameters()
        loss, n_samples, metrics = client.evaluate(params, {})
        assert isinstance(loss, float)
        assert n_samples > 0
        assert "rmse" in metrics


# ── FedAvg ──────────────────────────────────────────────────────────────────

def _fedavg(client_params):
    total = sum(n for _, n in client_params)
    averaged = [np.zeros_like(client_params[0][0][i])
                for i in range(len(client_params[0][0]))]
    for params, n in client_params:
        w = n / total
        for i, p in enumerate(params):
            averaged[i] += p * w
    return averaged


class TestFedAvg:

    def test_equal_weights_is_mean(self):
        a = [np.array([1.0, 2.0]), np.array([3.0])]
        b = [np.array([3.0, 4.0]), np.array([5.0])]
        result = _fedavg([(a, 10), (b, 10)])
        np.testing.assert_allclose(result[0], [2.0, 3.0])
        np.testing.assert_allclose(result[1], [4.0])

    def test_weighted_average(self):
        a = [np.array([0.0])]
        b = [np.array([10.0])]
        result = _fedavg([(a, 1), (b, 3)])
        np.testing.assert_allclose(result[0], [7.5])


# ── Ronda FL completa ──────────────────────────────────────────────────────

class TestFederatedRound:

    def test_one_round_reduces_loss(self, split, device):
        """Una ronda FL (fit + aggregate) debe reducir la loss vs el modelo sin entrenar."""
        model_c = get_model("rnn", hidden_dim=16, dropout=0.0).to(device)
        model_d = get_model("rnn", hidden_dim=16, dropout=0.0).to(device)

        init_state = copy.deepcopy(model_c.state_dict())
        model_d.load_state_dict(init_state)

        client_c = _make_client("hospital_c", split, model_c, device, local_epochs=3)
        client_d = _make_client("hospital_d", split, model_d, device, local_epochs=3)

        global_params = model_c.get_parameters()

        val_loader = DataLoader(
            split.val_combined, batch_size=16, shuffle=False,
        )
        eval_model = get_model("rnn", hidden_dim=16, dropout=0.0).to(device)
        eval_model.load_state_dict(init_state)
        loss_before, _ = evaluate(eval_model, val_loader, device)

        params_c, n_c, _ = client_c.fit(global_params, {})
        params_d, n_d, _ = client_d.fit(global_params, {})
        aggregated = _fedavg([(params_c, n_c), (params_d, n_d)])

        eval_model.set_parameters(aggregated)
        loss_after, _ = evaluate(eval_model, val_loader, device)

        assert loss_after < loss_before, (
            f"La loss debería bajar tras una ronda FL: {loss_before:.6f} -> {loss_after:.6f}"
        )


# ── Config ──────────────────────────────────────────────────────────────────

class TestFederationConfig:

    def test_default_values(self):
        cfg = FederationConfig()
        assert cfg.num_rounds == 10
        assert cfg.local_epochs == 3

    def test_from_toml(self, tmp_path):
        toml_content = """
name = "test_fl"
skill = "cutting"
profiles = ["hospital_c", "hospital_d"]

[federation]
num_rounds = 5
local_epochs = 2
"""
        p = tmp_path / "config.toml"
        p.write_text(toml_content)
        cfg = ExperimentConfig.from_toml(p)
        assert cfg.federation.num_rounds == 5
        assert cfg.federation.local_epochs == 2

    def test_from_toml_defaults_when_missing(self, tmp_path):
        toml_content = """
name = "test_no_fl"
skill = "cutting"
"""
        p = tmp_path / "config.toml"
        p.write_text(toml_content)
        cfg = ExperimentConfig.from_toml(p)
        assert cfg.federation.num_rounds == 10
        assert cfg.federation.local_epochs == 3
