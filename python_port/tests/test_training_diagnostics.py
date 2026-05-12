"""
训练诊断与评估口径优化的专项测试。

这些测试不启动完整训练，只验证本次新增的配置推导、指标解析和日志诊断逻辑。
"""

from __future__ import annotations

import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_derive_eval_files_from_train_files():
    from train_ppo_3 import _derive_eval_files_from_train_files

    train_files = [
        "1-2-1.txt",
        "1-2-2.txt",
        r"D:\nets\3-1-8.txt",
        "bad-name.txt",
    ]
    got = _derive_eval_files_from_train_files(train_files, suffix="20")
    assert got == ["1-2-20.txt", "3-1-20.txt"], got
    print("  [PASS] test_derive_eval_files_from_train_files")


def test_parse_suite_summary():
    from train_ppo_3 import _parse_suite_summary

    got = _parse_suite_summary("seen_pool_summary=success:3/4,avg_makespan:120,worst_makespan:300")
    assert got["success"] == 3
    assert got["total"] == 4
    assert abs(got["success_rate"] - 0.75) < 1e-12
    assert got["avg_makespan"] == 120.0
    assert got["worst_makespan"] == 300.0
    print("  [PASS] test_parse_suite_summary")


def test_build_epoch_diagnostics_flags_anomalies():
    from train_ppo_3 import _build_epoch_diagnostics

    previous = {"avg_reward": 42.0, "train_loss": 0.10}
    metrics = {
        "avg_reward": -5.0,
        "train_loss": 0.145,
        "eval_makespan": 76000,
        "best_show": 32000,
        "kl": 0.001,
        "eval_pool_due": True,
        "eval_pool_configured": True,
        "eval_pool_metrics": None,
        "steps_collected": 100,
        "epoch_elapsed_sec": 2.0,
    }
    diagnostics = _build_epoch_diagnostics(metrics, previous_metrics=previous, target_kl=0.07)
    joined = " ".join(diagnostics)
    assert "eval_vs_best_gap=" in joined
    assert "low_kl=" in joined
    assert "eval_pool_due_but_missing" in joined
    assert "reward_swing=-47.00" in joined
    assert "loss_jump=+0.0450" in joined
    assert "throughput=50.0steps/s" in joined
    print("  [PASS] test_build_epoch_diagnostics_flags_anomalies")


def test_env_choice_parsing():
    from train_ppo_3 import _env_choice

    key = "_PYTEST_LR_SCHEDULE"
    os.environ[key] = "COSINE"
    assert _env_choice(key, "linear", {"linear", "cosine"}) == "cosine"
    os.environ[key] = "bad"
    assert _env_choice(key, "linear", {"linear", "cosine"}) == "linear"
    del os.environ[key]
    assert _env_choice(key, "cosine", {"linear", "cosine"}) == "cosine"
    print("  [PASS] test_env_choice_parsing")


def test_set_global_seed_is_repeatable():
    import random
    import numpy as np
    import torch
    from train_ppo_3 import _set_global_seed

    _set_global_seed(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    _set_global_seed(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
    assert first == second
    print("  [PASS] test_set_global_seed_is_repeatable")


def test_cosine_lr_schedule_decays_more_slowly_than_linear():
    from petri_gcn_ppo_4_1 import PetriNetGCNPPOPro

    linear = object.__new__(PetriNetGCNPPOPro)
    linear.initial_lr = 3e-4
    linear.lr_schedule = "linear"
    linear.lr_min_ratio = 1e-5 / 3e-4
    linear.lr_decay_horizon = 1.0

    cosine = object.__new__(PetriNetGCNPPOPro)
    cosine.initial_lr = 3e-4
    cosine.lr_schedule = "cosine"
    cosine.lr_min_ratio = 0.35
    cosine.lr_decay_horizon = 1.5

    before = [linear._scheduled_lr(p) for p in (0.0, 0.5, 1.0)]
    after = [cosine._scheduled_lr(p) for p in (0.0, 0.5, 1.0)]
    assert after[0] == before[0] == 3e-4
    assert after[1] > before[1], (before, after)
    assert after[2] > before[2], (before, after)
    assert after[2] > 1.5e-4, after
    print(f"  [PASS] test_cosine_lr_schedule_decays_more_slowly_than_linear before={before} after={after}")


def test_epoch_summary_writes_diagnostics():
    from train_ppo_3 import PetriNetGCNPPOProHQ

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "progress.txt")
        obj = object.__new__(PetriNetGCNPPOProHQ)
        obj.extra_info = {}
        obj.target_kl = 0.07
        obj.verbose = False
        obj.epoch_log_path = log_path
        obj._last_epoch_metrics = {"avg_reward": 40.0, "train_loss": 0.10}

        metrics = {
            "epoch_idx": 2,
            "env_name": "unit-test-env",
            "total_steps": 128,
            "max_train_steps": 256,
            "steps_collected": 64,
            "train_loss": 0.14,
            "actor_loss": -0.01,
            "critic_loss": 0.30,
            "avg_reward": 10.0,
            "eval_success": True,
            "eval_makespan": 80000,
            "eval_show": 80000,
            "best_show": 32000,
            "kl": 0.001,
            "learning_rate": 0.0001,
            "entropy_coef": 0.2,
            "temperature": 1.5,
            "eval_pool_due": True,
            "eval_pool_configured": True,
            "eval_pool_metrics": None,
            "pool_metrics": None,
            "epoch_elapsed_sec": 1.0,
        }
        obj._log_epoch_summary(metrics)

        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "Diagnostics|" in text
        assert "eval_vs_best_gap=" in text
        assert "low_kl=" in text
        assert "eval_pool_due_but_missing" in text
        assert "reward_swing=-30.00" in text
    print("  [PASS] test_epoch_summary_writes_diagnostics")


def main():
    tests = [
        test_derive_eval_files_from_train_files,
        test_parse_suite_summary,
        test_build_epoch_diagnostics_flags_anomalies,
        test_env_choice_parsing,
        test_set_global_seed_is_repeatable,
        test_cosine_lr_schedule_decays_more_slowly_than_linear,
        test_epoch_summary_writes_diagnostics,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except BaseException as exc:
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
    if failed:
        sys.exit(1)
    print("全部训练诊断专项测试通过")


if __name__ == "__main__":
    main()
