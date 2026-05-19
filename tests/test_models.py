"""Tests para models/ — interfaz Flower, registry, forward shapes."""
import numpy as np
import pytest
import torch

from surgical_fl.models.registry import get_model, list_models
from surgical_fl.models.base import SurgicalModel, ModelMetadata
from surgical_fl.models.trajectory.base import TrajectoryModel
from surgical_fl.models.trajectory.mlp import TrajectoryMLP


class TestTrajectoryMLP:

    def setup_method(self):
        self.model = TrajectoryMLP(input_dim=2, hidden_dim=32, output_dim=2)

    def test_inherits_from_surgical_model(self):
        assert isinstance(self.model, SurgicalModel)
        assert isinstance(self.model, TrajectoryModel)

    def test_forward_preserves_shape(self):
        x = torch.randn(8, 49, 2)        # batch, T, dims
        y = self.model(x)
        assert y.shape == x.shape

    def test_forward_works_without_time_dim(self):
        x = torch.randn(8, 2)             # batch, dims (sin T)
        y = self.model(x)
        assert y.shape == x.shape

    def test_metadata(self):
        meta = self.model.metadata
        assert isinstance(meta, ModelMetadata)
        assert meta.task == "trajectory_prediction"
        assert meta.input_shape == (2,)
        assert meta.output_shape == (2,)


class TestSerializableInterface:
    """Valida el contrato con Flower: serialización de pesos."""

    def setup_method(self):
        self.model = TrajectoryMLP(input_dim=2, hidden_dim=16)

    def test_get_parameters_returns_numpy(self):
        params = self.model.get_parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters_restores_weights(self):
        original = self.model.get_parameters()
        # Modificar pesos
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()
        # Restaurar
        self.model.set_parameters(original)
        restored = self.model.get_parameters()
        for o, r in zip(original, restored):
            np.testing.assert_array_equal(o, r)

    def test_two_models_can_sync_weights(self):
        """Simulación: server envía pesos, cliente los carga."""
        server_model = TrajectoryMLP(input_dim=2, hidden_dim=16)
        client_model = TrajectoryMLP(input_dim=2, hidden_dim=16)
        # Sincronizar pesos
        client_model.set_parameters(server_model.get_parameters())
        # eval() desactiva dropout para que la inferencia sea determinista
        server_model.eval()
        client_model.eval()
        x = torch.randn(4, 2)
        torch.testing.assert_close(server_model(x), client_model(x))


class TestRegistry:

    def test_mlp_registered(self):
        assert "mlp" in list_models()

    def test_get_model_returns_instance(self):
        model = get_model("mlp")
        assert isinstance(model, SurgicalModel)

    def test_get_model_passes_kwargs(self):
        model = get_model("mlp", hidden_dim=128, dropout=0.3)
        assert model.hidden_dim == 128

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="no encontrado"):
            get_model("xgboost")
"""Tests para models/ — interfaz Flower, registry, forward shapes."""
import numpy as np
import pytest
import torch

from surgical_fl.models.registry import get_model, list_models
from surgical_fl.models.base import SurgicalModel, ModelMetadata
from surgical_fl.models.trajectory.base import TrajectoryModel
from surgical_fl.models.trajectory.mlp import TrajectoryMLP


class TestTrajectoryMLP:

    def setup_method(self):
        self.model = TrajectoryMLP(input_dim=2, hidden_dim=32, output_dim=2)

    def test_inherits_from_surgical_model(self):
        assert isinstance(self.model, SurgicalModel)
        assert isinstance(self.model, TrajectoryModel)

    def test_forward_preserves_shape(self):
        x = torch.randn(8, 49, 2)        # batch, T, dims
        y = self.model(x)
        assert y.shape == x.shape

    def test_forward_works_without_time_dim(self):
        x = torch.randn(8, 2)             # batch, dims (sin T)
        y = self.model(x)
        assert y.shape == x.shape

    def test_metadata(self):
        meta = self.model.metadata
        assert isinstance(meta, ModelMetadata)
        assert meta.task == "trajectory_prediction"
        assert meta.input_shape == (2,)
        assert meta.output_shape == (2,)


class TestSerializableParameters:
    """Valida la serialización de pesos como lista de arrays numpy."""

    def setup_method(self):
        self.model = TrajectoryMLP(input_dim=2, hidden_dim=16)

    def test_get_parameters_returns_numpy(self):
        params = self.model.get_parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_parameters_restores_weights(self):
        original = self.model.get_parameters()
        # Modificar pesos
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()
        # Restaurar
        self.model.set_parameters(original)
        restored = self.model.get_parameters()
        for o, r in zip(original, restored):
            np.testing.assert_array_equal(o, r)

    def test_two_models_can_sync_weights(self):
        """Un modelo puede copiar los pesos de otro vía serialización numpy."""
        model_a = TrajectoryMLP(input_dim=2, hidden_dim=16)
        model_b = TrajectoryMLP(input_dim=2, hidden_dim=16)
        # Sincronizar pesos
        model_b.set_parameters(model_a.get_parameters())
        # eval() desactiva dropout para que la inferencia sea determinista
        model_a.eval()
        model_b.eval()
        x = torch.randn(4, 2)
        torch.testing.assert_close(model_a(x), model_b(x))


class TestRegistry:

    def test_mlp_registered(self):
        assert "mlp" in list_models()

    def test_get_model_returns_instance(self):
        model = get_model("mlp")
        assert isinstance(model, SurgicalModel)

    def test_get_model_passes_kwargs(self):
        model = get_model("mlp", hidden_dim=128, dropout=0.3)
        assert model.hidden_dim == 128

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="no encontrado"):
            get_model("xgboost")