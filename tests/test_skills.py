"""Tests para domain/skills/"""
import numpy as np
from surgical_fl.domain.skills.cutting import CuttingSkill
from surgical_fl.data.generators.factory import build_generator


class TestCuttingSkill:

    def setup_method(self):
        self.skill = CuttingSkill()

    def test_name(self):
        assert self.skill.name == "cutting"

    def test_constraints_exist(self):
        c = self.skill.constraints
        assert c.min_length > 0
        assert c.max_curvature > 0

    def test_evaluate_returns_expected_keys(self):
        gen = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        traj = gen.generate_one()
        metrics = self.skill.evaluate_trajectory(traj)
        assert "task_score" in metrics
        assert "smoothness" in metrics
        assert "path_error" in metrics
        assert "length" in metrics

    def test_score_between_zero_and_one(self):
        gen = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        for traj in gen.generate(20):
            score = self.skill.evaluate_trajectory(traj)["task_score"]
            assert 0.0 <= score <= 1.0

    def test_expert_scores_higher_than_noisy(self):
        """Hospital A (experto) debería puntuar mejor que Hospital B."""
        gen_a = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        gen_b = build_generator("cutting", "hospital_b", trajectory_length=50, seed=0)
        scores_a = [self.skill.evaluate_trajectory(t)["task_score"]
                     for t in gen_a.generate(30)]
        scores_b = [self.skill.evaluate_trajectory(t)["task_score"]
                     for t in gen_b.generate(30)]
        assert np.mean(scores_a) > np.mean(scores_b)

    def test_degenerate_trajectory(self):
        """Trayectoria de un solo punto debe devolver score 0."""
        traj = np.array([[0.5, 0.0]])
        metrics = self.skill.evaluate_trajectory(traj)
        assert metrics["task_score"] == 0.0

    def test_is_valid(self):
        gen = build_generator("cutting", "hospital_a", trajectory_length=50, seed=0)
        traj = gen.generate_one()
        assert self.skill.is_valid(traj)

    def test_is_valid_rejects_bad_shape(self):
        bad = np.zeros((10, 3))  # 3D en vez de 2D
        assert not self.skill.is_valid(bad)