from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_JACKKNIFE_PATH = Path(__file__).resolve().parents[1] / "science_runner" / "jackknife.py"
_SPEC = importlib.util.spec_from_file_location("cosmos2_science_runner_jackknife", _JACKKNIFE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)  # type: ignore[union-attr]

JackknifeConfig = _MODULE.JackknifeConfig
JackknifeResampler = _MODULE.JackknifeResampler


def test_jackknife_masks_deterministic_for_fixed_seed() -> None:
    cfg = JackknifeConfig(
        enabled=True,
        n_draws=5,
        fraction_removed=0.2,
        random_seed=123,
        datasets_to_test=["sn", "bao"],
    )
    resampler_a = JackknifeResampler(cfg, ["sn", "bao"])
    resampler_b = JackknifeResampler(cfg, ["sn", "bao"])

    resampler_a.set_dataset_size("sn", 10)
    resampler_a.set_dataset_size("bao", 7)
    resampler_b.set_dataset_size("sn", 10)
    resampler_b.set_dataset_size("bao", 7)

    masks_a = resampler_a.generate_masks()
    masks_b = resampler_b.generate_masks()

    assert len(masks_a) == len(masks_b)
    for mask_a, mask_b in zip(masks_a, masks_b, strict=True):
        assert mask_a.random_seed == mask_b.random_seed
        assert set(mask_a.dataset_masks.keys()) == set(mask_b.dataset_masks.keys())
        for name in mask_a.dataset_masks:
            assert np.array_equal(mask_a.dataset_masks[name], mask_b.dataset_masks[name])


def test_jackknife_config_from_dict_coerces_seed() -> None:
    cfg = JackknifeConfig.from_dict({"enabled": True, "random_seed": "42"})
    assert cfg.enabled is True
    assert cfg.random_seed == 42
