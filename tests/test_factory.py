"""Tests para data/generators/factory.py"""
import numpy as np
import pytest
from surgical_fl.data.generators.factory import (
    build_generator,
    list_available_combinations,
)


class TestFactory:

    def test_all_combinations_produce_valid_data(self):
        for skill, profile_name in list_available_combinations():
            gen = build_generator(skill=skill, profile_name=profile_name,
                                  trajectory_length=20, seed=42)
            data = gen.generate(3)
            assert data.shape == (3, 20, 2), f"Fallo en ({skill}, {profile_name})"

    def test_minimum_combinations_exist(self):
        combos = list_available_combinations()
        assert len(combos) >= 2

    def test_unknown_skill_raises(self):
        with pytest.raises(ValueError, match="no tiene selector"):
            build_generator("nonexistent_skill", "hospital_a")

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="no encontrado"):
            build_generator("cutting", "hospital_z")


class TestHeterogeneity:

    def test_hospital_b_noisier_than_a(self):
        gen_a = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        gen_b = build_generator("cutting", "hospital_b", trajectory_length=50, seed=0)
        data_a = gen_a.generate(100)
        data_b = gen_b.generate(100)
        std_y_a = np.std(data_a[:, :, 1])
        std_y_b = np.std(data_b[:, :, 1])
        assert std_y_b > std_y_a

    def test_different_hospitals_different_generator_class(self):
        gen_a = build_generator("cutting", "hospital_a", seed=0)
        gen_b = build_generator("cutting", "hospital_b", seed=0)
        gen_c = build_generator("cutting", "hospital_c", seed=0)
        assert type(gen_a).__name__ != type(gen_b).__name__ != type(gen_c).__name__

    def test_hospital_c_more_curved_than_a(self):
        """Hospital C (spline en S) se aparta más de la recta que Hospital A.

        Se promedia sobre muchas muestras para cancelar el ruido y medir la
        FORMA de la incisión, no el jitter (un trazo recto ruidoso puede tener
        más segundas diferencias que una curva suave).
        """
        gen_a = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        gen_c = build_generator("cutting", "hospital_c", trajectory_length=50, seed=0)
        mean_y_a = gen_a.generate(200)[:, :, 1].mean(axis=0)
        mean_y_c = gen_c.generate(200)[:, :, 1].mean(axis=0)
        amplitude_a = mean_y_a.max() - mean_y_a.min()
        amplitude_c = mean_y_c.max() - mean_y_c.min()
        assert amplitude_c > amplitude_a