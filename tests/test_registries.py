"""Tests para los tres registries simétricos."""
import pytest
from surgical_fl.domain.skills.registry import get_skill, list_skills
from surgical_fl.domain.profiles.registry import get_profile, list_profiles
from surgical_fl.data.generators.registry import (
    get_generator_classes,
    list_skills_with_generators,
)


class TestSkillsRegistry:

    def test_cutting_exists(self):
        skill = get_skill("cutting")
        assert skill.name == "cutting"

    def test_list_not_empty(self):
        assert len(list_skills()) > 0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="no encontrada"):
            get_skill("teleportation")


class TestProfilesRegistry:

    def test_all_profiles_valid(self):
        for name in list_profiles():
            profile = get_profile(name)
            assert profile.noise_std > 0
            assert profile.name != ""

    def test_list_not_empty(self):
        assert len(list_profiles()) >= 2

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="no encontrado"):
            get_profile("hospital_xyz")


class TestGeneratorRegistry:

    def test_cutting_has_generators(self):
        classes = get_generator_classes("cutting")
        assert len(classes) >= 2

    def test_list_not_empty(self):
        assert "cutting" in list_skills_with_generators()

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="no tiene generadores"):
            get_generator_classes("flying")
