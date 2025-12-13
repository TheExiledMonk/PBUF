"""Basic tests for the prediction modules."""

from typing import Sequence

import json
import math
import numpy as np

from cosmos2.models.lcdm.model import LCDMModel
from cosmos2.parameters.central_authority import MPC_TO_KM, SECONDS_PER_GYR
from cosmos2.predictions import run_prediction_for_model


def test_growth_prediction_runs() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model("growth", lcdm, {"zmax": 2.0, "points": 20})
    assert result.status == "success"
    assert "f_at_z0" in result.results
    assert result.results["f_at_z0"] > 0.0


def test_g_effective_prediction_matches_gr() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model("g-effective", lcdm, {"points": 120})
    assert result.status == "success"
    prediction = result.results
    z_vals = np.asarray(prediction["z"], dtype=float)
    a_vals = np.asarray(prediction["a"], dtype=float)
    mu_vals = np.asarray(prediction["mu"], dtype=float)
    mask_valid = np.asarray(prediction["mask_valid"], dtype=bool)

    assert len(z_vals) == len(a_vals) == len(mu_vals) == len(mask_valid)
    assert mask_valid.sum() >= 10

    low_z_mask = mask_valid & (z_vals <= 1.0)
    assert low_z_mask.sum() > 0
    np.testing.assert_allclose(mu_vals[low_z_mask], np.ones_like(mu_vals[low_z_mask]), rtol=5e-2, atol=5e-2)

    summary = prediction["summary"]
    assert summary["mu0"] is not None and math.isclose(summary["mu0"], 1.0, rel_tol=5e-2, abs_tol=5e-2)
    assert summary["mu_z0p5"] is not None and math.isclose(summary["mu_z0p5"], 1.0, rel_tol=5e-2, abs_tol=5e-2)
    assert summary["mu_z1"] is not None and math.isclose(summary["mu_z1"], 1.0, rel_tol=5e-2, abs_tol=5e-2)
    assert summary["mu_mean_0_1"] is not None
    assert math.isclose(summary["mu_mean_0_1"], 1.0, rel_tol=1e-3, abs_tol=1e-3)

    assert "description" in prediction["meta"]
    assert "μ(z)" in prediction["meta"]["description"]
    assert result.plots
    json.dumps(result.to_dict())


def test_redshift_drift_prediction_basics() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model("redshift-drift", lcdm, {"points": 20})
    assert result.status == "success"
    assert result.results["dzdt0_z2"] is not None
    assert result.results["dv10_z3"] is not None
    assert "model" in result.metadata


def test_gw_propagation_prediction_defaults_and_summaries() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model(
        "gw-propagation",
        lcdm,
        {"zmax": 2.0, "points": 40, "z_key": "0.1,1.0,2.0"},
    )
    assert result.status == "success"
    assert result.results["zmax"] == 2.0
    assert result.results["z_keys"] == [0.1, 1.0, 2.0]
    rd_values = result.results["RD_at_z"]
    assert len(rd_values) == 3
    assert all(math.isclose(val, 1.0, rel_tol=1e-8) for val in rd_values)
    delta_t = result.results["Delta_t_GW_EM_at_z"]
    assert all(math.isclose(val, 0.0, abs_tol=1e-8) for val in delta_t)
    assert result.metadata["used_wave_speed"] is False
    assert result.results["DL_EM_at_z"][0] > 0.0


def test_redshift_drift_table_and_lcdm_reference() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model(
        "redshift-drift",
        lcdm,
        {
            "points": 20,
            "output_table": True,
            "output_plot": True,
            "compare_lcdm": True,
        },
    )
    assert result.status == "success"
    assert result.tables
    assert result.tables[0].name == "redshift_drift_vs_z"
    assert len(result.plots) == 2
    ratio = result.results["ratio_dv10_z3"]
    assert ratio is not None
    assert math.isclose(ratio, 1.0, rel_tol=1e-6)
    assert result.metadata["compare_lcdm"] is True


def test_growth_prediction_with_sigma8_includes_table() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model(
        "growth",
        lcdm,
        {"zmax": 3.0, "points": 30, "include_s8": True, "output_table": True},
    )
    assert result.status == "success"
    assert "s8_today" in result.results
    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.name == "growth_vs_z"


def test_acceleration_onset_prediction_basic_ldm() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model(
        "acceleration-onset",
        lcdm,
        {"zmax": 3.0, "points": 200, "output_table": True, "output_plot": True},
    )
    assert result.status == "success"
    assert "z_accel" in result.results
    z_accel = result.results["z_accel"]
    assert isinstance(z_accel, float)
    assert 0.2 < z_accel < 1.2
    assert "q0" in result.results
    assert result.results["q0"] < 0.0
    assert len(result.tables) == 1
    assert result.tables[0].name == "q_vs_z"
    assert len(result.plots) == 1
    assert result.plots[0].name == "q_vs_z_plot"


class _FakeCurvatureModel:
    def __init__(
        self,
        *,
        alpha: float | None = None,
        Omega_b0: float | None = None,
        k_sat: float | None = None,
        k_max: float | None = None,
        epsilon0: float = 1.0,
    ) -> None:
        self.parameters = {}
        if alpha is not None:
            self.parameters["alpha"] = alpha
        if Omega_b0 is not None:
            self.parameters["Omega_b0"] = Omega_b0
        if k_sat is not None:
            self.parameters["k_sat"] = k_sat
        if k_max is not None:
            self.parameters["k_max"] = k_max
        self._epsilon0 = epsilon0

    def elastic_stiffness(self, a: float | list[float]) -> np.ndarray:
        arr = np.asarray(a, dtype=float)
        return np.full(arr.shape, self._epsilon0, dtype=float)


def test_curvature_identity_prediction_requires_alpha_and_omega_b0() -> None:
    model = _FakeCurvatureModel()
    result = run_prediction_for_model("curvature-identity", model, {})
    assert result.status == "error"
    assert result.metadata.get("error") == "missing_alpha_or_Omega_b0"


def test_curvature_identity_prediction_outputs_table_and_plot() -> None:
    model = _FakeCurvatureModel(
        alpha=0.02,
        Omega_b0=0.04,
        k_sat=0.97,
        k_max=0.4,
        epsilon0=0.42,
    )
    result = run_prediction_for_model(
        "curvature-identity",
        model,
        {"output_table": True, "output_plot": True},
    )
    assert result.status == "success"
    assert result.metadata.get("has_k_max") is True
    assert math.isclose(result.results["Omega_b0_pred"], 0.04, rel_tol=1e-6)
    assert math.isclose(result.results["k_sat_pred"], 0.98, rel_tol=1e-6)
    assert len(result.tables) == 1
    assert result.tables[0].name == "curvature_identity_components"
    assert len(result.tables[0].rows) == 3
    assert result.plots
    assert result.plots[0].name == "curvature_identity_bar"


class _SimpleGrowthModel:
    def __init__(self, *, alpha: float = 0.02) -> None:
        self.parameters = {
            "H0": 70.0,
            "Omega_m0": 0.3,
            "alpha": alpha,
        }

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        H0 = float(self.parameters["H0"])
        Om0 = float(self.parameters["Omega_m0"])
        values = H0 * np.sqrt(np.clip(Om0 * (1.0 + arr) ** 3 + (1.0 - Om0), 0.0, np.inf))
        return float(values[0]) if np.isscalar(z) else values

    def growth_factor(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        values = np.exp(-0.2 * arr)
        return float(values[0]) if np.isscalar(z) else values


class _ToyLCDMModel:
    def __init__(self, *, H0: float = 70.0, Omega_m0: float = 0.3) -> None:
        self.parameters = {"H0": float(H0), "Omega_m0": float(Omega_m0)}
        self._H0 = float(H0)
        self._Omega_m0 = float(Omega_m0)

    def H(self, a: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(a, dtype=float))
        values = self._Omega_m0 / arr**3 + (1.0 - self._Omega_m0)
        values = np.clip(values, 0.0, np.inf)
        result = self._H0 * np.sqrt(values)
        return float(result[0]) if np.isscalar(a) else result

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        a = 1.0 / (1.0 + arr)
        return self.H(a)

    def comoving_distance(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        return float(arr[0]) if np.isscalar(z) else arr


class _ToyElasticSurface:
    def omega_sigma(self, a: float | Sequence[float]) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(a, dtype=float))
        return 0.1 * arr


class _ToyElasticModel:
    def __init__(self, *, H0: float = 70.0) -> None:
        self.parameters = {"H0": float(H0)}
        self._H0 = float(H0)
        self._elastic_surface = _ToyElasticSurface()

    @property
    def elastic(self) -> _ToyElasticSurface:
        return self._elastic_surface

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        a = 1.0 / (1.0 + arr)
        values = self._H0 * (1.0 + a)
        return float(values[0]) if np.isscalar(z) else values


def test_horizon_evolution_prediction_toy_model_behaviour() -> None:
    class _ToyHorizonModel:
        def __init__(self, *, H0: float = 70.0, Omega_m0: float = 0.3) -> None:
            self.parameters = {"H0": float(H0), "Omega_m0": float(Omega_m0)}
            self._H0 = float(H0)
            self._Omega_m0 = float(Omega_m0)

        def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
            arr = np.atleast_1d(np.asarray(z, dtype=float))
            values = self._H0 * np.sqrt(
                np.clip(self._Omega_m0 * (1.0 + arr) ** 3 + (1.0 - self._Omega_m0), 0.0, np.inf)
            )
            return float(values[0]) if np.isscalar(z) else values

    model = _ToyHorizonModel()
    result = run_prediction_for_model(
        "horizon-evolution",
        model,
        {"zmax": 6.0, "points": 201},
    )
    assert result.status == "success"
    assert "meta" in result.results
    assert result.results["meta"]["version"] == "1.0"
    assert result.results["meta"]["distance_unit"] == "same as c/H(z)"
    assert "Horizon evolution prediction" in result.results["meta"]["description"]

    results = result.results
    z_arr = np.asarray(results["z"], dtype=float)
    mask_valid = np.asarray(results["mask_valid"], dtype=bool)
    assert mask_valid.sum() >= 10
    R_phys = np.asarray(results["R_H_phys"], dtype=float)
    R_com = np.asarray(results["R_H_comoving"], dtype=float)
    chi = np.asarray(results["chi_particle"], dtype=float)

    R_phys_valid = R_phys[mask_valid]
    R_com_valid = R_com[mask_valid]
    assert np.all(np.isfinite(R_phys_valid))
    assert np.all(np.isfinite(R_com_valid))
    assert np.all(R_phys_valid > 0.0)
    assert np.all(R_com_valid > 0.0)
    assert R_phys_valid[0] > R_phys_valid[-1]

    assert np.max(R_com_valid) > R_com_valid[0]
    peak_index = int(np.argmax(R_com_valid))
    assert 0 < peak_index < len(R_com_valid) - 1

    chi_valid = chi[mask_valid]
    assert np.all(chi_valid >= 0.0)
    assert np.all(np.diff(chi_valid) <= 1e-6)

    summary = results["summary"]
    for key in ("R_H0_phys", "R_H0_comoving", "R_H_z1_comoving", "R_H_z6_comoving"):
        value = summary[key]
        assert value is not None
        assert math.isfinite(value)

    serialized = json.dumps(result.to_dict())
    assert "horizon-evolution" in serialized


class _ToyAPModel:
    def __init__(self, *, H0: float = 70.0) -> None:
        self.parameters = {"H0": float(H0)}
        self._H0 = float(H0)
        self._c = 299_792.458

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        values = np.full(arr.shape, self._H0, dtype=float)
        return float(values[0]) if np.isscalar(z) else values

    def DA(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        denom = np.clip(1.0 + arr, 1e-12, np.inf)
        values = (self._c / self._H0) * arr / denom
        return float(values[0]) if np.isscalar(z) else values


def test_q_curve_prediction_matches_toy_lcdm() -> None:
    model = _ToyLCDMModel()
    result = run_prediction_for_model("q-curve", model, {"zmin": 0.0, "zmax": 3.0, "points": 201})
    assert result.status == "success"
    prediction = result.results
    assert prediction["name"] == "q-curve"

    z_vals = np.asarray(prediction["z"], dtype=float)
    a_vals = np.asarray(prediction["a"], dtype=float)
    q_vals = np.asarray(prediction["q"], dtype=float)
    mask_valid = np.asarray(prediction["mask_valid"], dtype=bool)

    assert len(z_vals) == len(a_vals) == len(q_vals) == len(mask_valid)
    assert all(isinstance(flag, (bool, np.bool_)) for flag in mask_valid)
    assert prediction["meta"]["notes"] == "Deceleration parameter q(z) computed from E(a) with finite differences."
    assert prediction["meta"]["model_name"] == "_ToyLCDMModel"

    omega_m0 = model._Omega_m0
    omega_l0 = 1.0 - omega_m0
    a_cubed = np.clip(a_vals**3, 1e-12, np.inf)
    e_squared = np.clip(omega_m0 / a_cubed + omega_l0, 0.0, np.inf)
    expected_q = -1.0 + (1.5 * omega_m0) / (a_cubed * e_squared)

    np.testing.assert_allclose(q_vals[mask_valid], expected_q[mask_valid], rtol=1e-3, atol=6e-4)

    summary_z_acc = prediction["summary"]["z_acc"]
    assert summary_z_acc is not None
    a_acc = (omega_m0 / (2.0 * omega_l0)) ** (1.0 / 3.0)
    expected_z_acc = (1.0 / a_acc) - 1.0
    assert math.isclose(summary_z_acc, expected_z_acc, rel_tol=1e-3)

    json.dumps(prediction)


def test_jerk_prediction_matches_lcdm_toy() -> None:
    model = _ToyLCDMModel()
    result = run_prediction_for_model("jerk", model, {"zmax": 2.0, "points": 150})
    assert result.status == "success"
    prediction = result.results
    assert prediction["name"] == "jerk"

    z_vals = np.asarray(prediction["z"], dtype=float)
    a_vals = np.asarray(prediction["a"], dtype=float)
    j_vals = np.asarray(prediction["j"], dtype=float)
    mask_valid = np.asarray(prediction["mask_valid"], dtype=bool)

    assert len(z_vals) == len(a_vals) == len(j_vals) == len(mask_valid)
    assert mask_valid.dtype == bool
    assert np.count_nonzero(mask_valid) > 0

    valid_j = j_vals[mask_valid]
    assert np.all(np.isfinite(valid_j))
    np.testing.assert_allclose(valid_j, np.ones_like(valid_j), rtol=1e-4, atol=1e-4)

    summary = prediction["summary"]
    assert summary["j0"] is not None
    assert summary["j_mean_0_1"] is not None
    assert math.isclose(summary["j0"], 1.0, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(summary["j_mean_0_1"], 1.0, rel_tol=1e-4, abs_tol=1e-4)


def test_statefinder_prediction_matches_lcdm_toy() -> None:
    model = _ToyLCDMModel()
    result = run_prediction_for_model("statefinder", model, {"zmax": 2.0, "points": 120})
    assert result.status == "success"
    prediction = result.results
    assert prediction["name"] == "statefinder"

    z_vals = np.asarray(prediction["z"], dtype=float)
    a_vals = np.asarray(prediction["a"], dtype=float)
    r_vals = np.asarray(prediction["r"], dtype=float)
    s_vals = np.asarray(prediction["s"], dtype=float)
    q_vals = np.asarray(prediction["q"], dtype=float)
    mask_valid = np.asarray(prediction["mask_valid"], dtype=bool)

    assert (
        len(z_vals)
        == len(a_vals)
        == len(r_vals)
        == len(s_vals)
        == len(q_vals)
        == len(mask_valid)
    )
    assert mask_valid.dtype == bool

    valid_mask = mask_valid & np.isfinite(r_vals) & np.isfinite(s_vals)
    assert np.count_nonzero(valid_mask) > 0

    np.testing.assert_allclose(r_vals[valid_mask], np.ones_like(r_vals[valid_mask]), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(s_vals[valid_mask], np.zeros_like(s_vals[valid_mask]), rtol=1e-4, atol=1e-4)

    summary = prediction["summary"]
    assert summary["r0"] is not None
    assert summary["s0"] is not None
    assert math.isclose(summary["r0"], 1.0, rel_tol=1e-4, abs_tol=1e-4)
    assert math.isclose(summary["s0"], 0.0, rel_tol=1e-4, abs_tol=1e-4)

    assert prediction["meta"]["notes"] == "Statefinder diagnostics r(z), s(z) computed from E(a) and its derivatives."
    assert "description" in prediction["meta"]
    assert prediction["meta"]["model_name"] == "_ToyLCDMModel"

    json.dumps(prediction)


def test_elastic_fraction_prediction_analytic_shape() -> None:
    model = _ToyElasticModel()
    config = {"z_min": 0.0, "z_max": 6.0, "points": 400}
    result = run_prediction_for_model("elastic-fraction", model, config)
    assert result.status == "success"
    payload = result.results
    assert payload["name"] == "elastic-fraction"

    z_vals = np.asarray(payload["z"], dtype=float)
    a_vals = np.asarray(payload["a"], dtype=float)
    f_vals = np.asarray(payload["f_sigma"], dtype=float)
    mask_valid = np.asarray(payload["mask_valid"], dtype=bool)
    expected_E = 1.0 + a_vals
    expected_ratio = 0.1 * a_vals / np.square(expected_E)
    np.testing.assert_allclose(f_vals[mask_valid], expected_ratio[mask_valid], rtol=1e-6)

    summary = payload["summary"]
    assert math.isclose(summary["Omega_sigma_0"], 0.1, rel_tol=1e-9)
    assert math.isclose(summary["f_sigma_0"], 0.025, rel_tol=1e-9)
    assert math.isclose(summary["f_sigma_peak"], 0.025, rel_tol=1e-9)
    assert math.isclose(summary["z_peak"], 0.0, abs_tol=1e-9)
    assert summary["z_half_peak_lo"] is None

    expected_half_a = 3.0 - 2.0 * math.sqrt(2.0)
    expected_half_z = (1.0 / expected_half_a) - 1.0
    assert summary["z_half_peak_hi"] is not None
    assert math.isclose(float(summary["z_half_peak_hi"]), expected_half_z, rel_tol=5e-3)

    assert payload["meta"]["version"] == "1.0"
    json.dumps(result.to_dict())


def test_void_size_prediction_requires_alpha() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model("void-size", lcdm, {"points": 5})
    assert result.status == "error"
    assert result.metadata.get("error") == "missing_alpha"


def test_void_size_prediction_with_lcdm_comparison() -> None:
    model = _SimpleGrowthModel(alpha=0.02)
    result = run_prediction_for_model(
        "void-size",
        model,
        {"zmax": 1.0, "points": 10, "compare_lcdm": True, "output_table": True, "output_plot": True},
    )
    assert result.status == "success"
    assert result.results["ratio_PBUF_over_LCDM_z0"] is not None
    assert result.tables
    assert result.tables[0].name == "void_radius_vs_z"
    assert len(result.plots) == 2


def test_void_size_prediction_without_reference_ratios() -> None:
    model = _SimpleGrowthModel(alpha=0.015)
    result = run_prediction_for_model("void-size", model, {"points": 6})
    assert result.status == "success"
    assert result.results["ratio_PBUF_over_LCDM_z0"] is None


def test_isw_cross_prediction_basic_lcdm() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model(
        "isw-cross",
        lcdm,
        {"zmin": 0.0, "zmax": 1.0, "points": 50},
    )
    assert result.status == "success"
    assert math.isfinite(result.results["A_ISW_PBUF"])
    assert result.results["A_ISW_LCDM"] is None
    assert result.results["ratio_PBUF_over_LCDM"] is None


def test_isw_cross_prediction_with_lcdm_comparison() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model(
        "isw-cross",
        lcdm,
        {"zmin": 0.0, "zmax": 1.0, "points": 60, "compare_lcdm": True, "output_table": True, "output_plot": True},
    )
    assert result.status == "success"
    assert result.tables
    assert result.tables[0].name == "isw_kernel_vs_z"
    assert len(result.plots) == 2
    assert math.isclose(result.results["ratio_PBUF_over_LCDM"], 1.0, rel_tol=1e-6)


def test_growth_index_payload_schema() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model(
        "growth-index",
        lcdm,
        {"zmax": 1.5, "points": 20},
    )
    assert result.status == "success"
    payload = result.results
    assert payload["name"] == "growth_index"
    assert set(payload.keys()) == {"name", "z", "f", "gamma", "mask_valid", "meta"}
    z_vals = payload["z"]
    f_vals = payload["f"]
    mask = payload["mask_valid"]
    assert len(z_vals) == len(f_vals) == len(payload["gamma"]) == len(mask) == 20
    assert all(isinstance(entry, bool) for entry in mask)
    meta = payload["meta"]
    assert meta["version"] == "1.0"
    assert "Growth index γ(z)" in meta["description"]


def test_growth_index_gamma_at_z0_matches_lcdm() -> None:
    lcdm = LCDMModel()
    result = run_prediction_for_model("growth-index", lcdm, {"points": 120})
    payload = result.results
    assert payload["mask_valid"][0] is True
    gamma0 = payload["gamma"][0]
    assert 0.48 < gamma0 < 0.62
    assert all(math.isfinite(val) for val in payload["f"])


class _EdSModel:
    def __init__(self) -> None:
        self.parameters = {"H0": 67.4, "Omega_m0": 1.1}

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        H0 = float(self.parameters["H0"])
        values = H0 * (1.0 + arr) ** 1.5
        return float(values[0]) if np.isscalar(z) else values


def test_growth_index_mask_blocks_invalid_regions() -> None:
    model = _EdSModel()
    result = run_prediction_for_model("growth-index", model, {"points": 40})
    payload = result.results
    assert result.metadata.get("warnings")
    assert not any(payload["mask_valid"])
    assert all(math.isnan(val) for val in payload["gamma"])
    assert all(math.isfinite(val) for val in payload["f"])


class _MockLensingBackend:
    def __init__(self) -> None:
        self.backend_name = "mock-lensing"

    def compute_cmb_kappa(self, ell: Sequence[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(ell, dtype=float)
        return 1e-8 * np.exp(-arr / 500.0) + 1e-9


class _MockLensingModel:
    def __init__(self) -> None:
        self.parameters = {"H0": 67.4, "Omega_m0": 0.3, "sigma8_0": 0.8}
        self.lensing = _MockLensingBackend()


def test_cmb_lensing_prediction_basic() -> None:
    model = _MockLensingModel()
    result = run_prediction_for_model(
        "cmb-lensing",
        model,
        {"ell_min": 10.0, "ell_max": 400.0, "n_ell": 60},
    )
    assert result.status == "success"
    payload = result.results
    assert payload["name"] == "cmb_lensing"
    ell = payload["ell"]
    cl = payload["cl_kappa"]
    mask = payload["mask_valid"]
    assert len(ell) == len(cl) == len(mask) == 60
    assert all(mask)
    assert result.metadata["valid_points"] == 60
    assert math.isfinite(payload["summary"]["A_L_eff"])


def test_cmb_lensing_relative_amplitude_matches_reference() -> None:
    ell_min = 8.0
    ell_max = 1200.0
    n_ell = 45
    backend = _MockLensingBackend()
    ell_ref = np.linspace(ell_min, ell_max, n_ell, dtype=float)
    cl_ref = backend.compute_cmb_kappa(ell_ref)
    model = _MockLensingModel()
    result = run_prediction_for_model(
        "cmb-lensing",
        model,
        {
            "ell_min": ell_min,
            "ell_max": ell_max,
            "n_ell": n_ell,
            "reference_spectrum": {
                "ell": ell_ref.tolist(),
                "cl_kappa": cl_ref.tolist(),
            },
        },
    )
    summary = result.results["summary"]
    assert "A_L_rel" in summary
    assert math.isclose(summary["A_L_rel"], 1.0, rel_tol=1e-6)


def test_cmb_lensing_prediction_serializes_to_json() -> None:
    model = _MockLensingModel()
    result = run_prediction_for_model("cmb-lensing", model, {"n_ell": 20})
    json.dumps(result.results)


class _ToyEdSModel:
    def __init__(self, *, H0: float = 70.0, Omega_m0: float = 1.0, sigma8_0: float = 0.8) -> None:
        self.parameters = {"H0": float(H0), "Omega_m0": float(Omega_m0), "sigma8_0": float(sigma8_0)}
        self._H0 = float(H0)
        self._Omega_m0 = float(Omega_m0)
        self._sigma8 = float(sigma8_0)

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        values = self._H0 * (1.0 + arr) ** 1.5
        return float(values[0]) if np.isscalar(z) else values

    def omega_m0(self) -> float:
        return self._Omega_m0

    def sigma8_today(self) -> float:
        return self._sigma8


def test_fsigma8_prediction_toy_model_matches_analytical_growth() -> None:
    sigma8_0 = 0.85
    model = _ToyEdSModel(sigma8_0=sigma8_0)
    result = run_prediction_for_model("fsigma8", model, {"zmax": 1.0, "points": 60})
    assert result.status == "success"
    payload = result.results
    summary = payload["summary"]
    assert math.isclose(summary["fs8_z0"], sigma8_0, rel_tol=1e-4, abs_tol=1e-5)
    assert math.isclose(summary["fs8_z0p5"], sigma8_0 * 2.0 / 3.0, rel_tol=1e-3, abs_tol=5e-3)
    assert math.isclose(summary["fs8_z1"], sigma8_0 * 0.5, rel_tol=1e-3, abs_tol=5e-3)
    D_norm = np.asarray(payload["D_norm"], dtype=float)
    assert math.isclose(D_norm[0], 1.0, rel_tol=1e-6)
    mask_valid = np.asarray(payload["mask_valid"], dtype=bool)
    assert mask_valid.sum() > 0
    sigma8_z = np.asarray(payload["sigma8_z"], dtype=float)
    np.testing.assert_allclose(
        sigma8_z[mask_valid], sigma8_0 * D_norm[mask_valid], rtol=1e-6, atol=0
    )
    f_vals = np.asarray(payload["f"], dtype=float)
    np.testing.assert_allclose(f_vals[mask_valid], np.ones_like(f_vals[mask_valid]), rtol=1e-3)
    fs8_vals = np.asarray(payload["fs8"], dtype=float)
    np.testing.assert_allclose(
        fs8_vals[mask_valid], f_vals[mask_valid] * sigma8_z[mask_valid], rtol=1e-6
    )
    assert payload["meta"]["sigma8_0"] == sigma8_0
    assert payload["meta"]["model_name"] == "_ToyEdSModel"
    assert payload["meta"]["notes"] == "fσ8(z) computed from normalized growth D(a) and σ8_0."


def test_fsigma8_prediction_payload_serializes() -> None:
    model = _ToyEdSModel()
    result = run_prediction_for_model("fsigma8", model, {"points": 10})
    json.dumps(result.results)


def test_ap_distortion_prediction_toy_model() -> None:
    model = _ToyAPModel()
    config = {"zmin": 0.0, "zmax": 3.0, "points": 301}
    result = run_prediction_for_model("ap-distortion", model, config)
    assert result.status == "success"

    payload = result.results
    assert payload["name"] == "ap-distortion"
    z_vals = np.asarray(payload["z"], dtype=float)
    H_vals = np.asarray(payload["H"], dtype=float)
    D_A_vals = np.asarray(payload["D_A"], dtype=float)
    F_vals = np.asarray(payload["F"], dtype=float)
    mask_valid = np.asarray(payload["mask_valid"], dtype=bool)

    assert len(z_vals) == len(H_vals) == len(D_A_vals) == len(F_vals) == len(mask_valid) == config["points"]
    assert mask_valid.dtype == bool
    assert np.count_nonzero(mask_valid) == len(z_vals) - 1
    assert np.all(np.isfinite(H_vals))
    assert np.all(np.isfinite(D_A_vals))
    assert np.all(np.isfinite(F_vals[mask_valid]))
    np.testing.assert_allclose(F_vals[mask_valid], z_vals[mask_valid], rtol=1e-9, atol=0)

    summary = payload["summary"]
    assert math.isclose(summary["F_z0p5"], 0.5, rel_tol=1e-9)
    assert math.isclose(summary["F_z1"], 1.0, rel_tol=1e-9)
    assert math.isclose(summary["F_z2"], 2.0, rel_tol=1e-9)

    meta = payload["meta"]
    assert meta["model_name"] == "_ToyAPModel"
    assert math.isfinite(meta["c"])
    assert "Alcock–Paczynski distortion parameter" in meta["description"]
    assert meta["version"] == "1.0"

    json.dumps(payload)


class _ToyLookbackModel:
    def __init__(self, *, H0: float = 70.0) -> None:
        self.parameters = {"H0": float(H0)}
        self._H0 = float(H0)

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        values = self._H0 * (1.0 + arr)
        return float(values[0]) if np.isscalar(z) else values


def test_lookback_prediction_monotonic_and_summary() -> None:
    model = _ToyLookbackModel(H0=70.0)
    result = run_prediction_for_model(
        "lookback",
        model,
        {"zmax": 6.0, "points": 120},
    )
    assert result.status == "success"

    z = np.asarray(result.results["z"], dtype=float)
    tL = np.asarray(result.results["tL_Gyr"], dtype=float)
    t_age = np.asarray(result.results["t_age_Gyr"], dtype=float)
    mask_valid = np.asarray(result.results["mask_valid"], dtype=bool)

    valid_count = int(np.count_nonzero(mask_valid))
    assert valid_count >= 2
    assert z.size == mask_valid.size == tL.size == t_age.size

    if valid_count > 1:
        assert np.all(np.diff(tL[mask_valid]) >= -1e-9)
        assert np.all(np.diff(t_age[mask_valid]) <= 1e-9)

    t0_expected = (MPC_TO_KM / SECONDS_PER_GYR) / 70.0
    assert math.isclose(result.results["t0_Gyr"], t0_expected, rel_tol=1e-6)
    assert math.isclose(result.metadata["summary"]["t0_Gyr"], t0_expected, rel_tol=1e-6)

    assert result.results["t_z1_Gyr"] is not None
    assert math.isfinite(float(result.results["t_z1_Gyr"]))
    assert result.results["t_z6_Gyr"] is not None
    assert math.isfinite(float(result.results["t_z6_Gyr"]))

    json.dumps(result.to_dict())


def test_age_z_prediction_behaves_like_cosmic_age_curve() -> None:
    model = _ToyLCDMModel()
    config = {"zmin": 0.0, "zmax": 10.0, "points": 300, "output_plot": True, "output_table": False}
    result = run_prediction_for_model("age-z", model, config)
    assert result.status == "success"

    payload = result.results
    assert payload["name"] == "age-z"
    z_values = np.asarray(payload["z"], dtype=float)
    t_age = np.asarray(payload["t_age_Gyr"], dtype=float)
    mask_valid = np.asarray(payload["mask_valid"], dtype=bool)
    summary = payload["summary"]

    assert bool(mask_valid[0])
    valid_count = int(np.count_nonzero(mask_valid))
    assert valid_count >= 10

    valid_ages = t_age[mask_valid]
    assert valid_ages.size == valid_count
    assert np.all(np.diff(valid_ages) <= 1e-6), "Cosmic age should decrease with redshift"
    assert math.isclose(valid_ages[0], summary["t0_Gyr"], rel_tol=1e-9)

    last_age = float(valid_ages[-1])
    assert abs(last_age) < 0.2

    t_z1 = summary["t_z1_Gyr"]
    t_z6 = summary["t_z6_Gyr"]
    assert t_z1 is not None and math.isfinite(t_z1) and t_z1 > 0.0
    assert t_z6 is not None and math.isfinite(t_z6) and 0.0 < t_z6 < t_z1

    half_redshift = summary["z_half_age"]
    assert half_redshift is not None
    assert 0.0 < float(half_redshift) < config["zmax"]

    assert "Cosmic age prediction" in payload["meta"]["description"]
    assert payload["meta"]["model_name"] == "_ToyLCDMModel"

    json.dumps(payload)


class _ToyPowerLawMatter:
    def __init__(self, *, amplitude: float = 1.0, slope: float = 1.0) -> None:
        self._amplitude = float(amplitude)
        self._slope = float(slope)

    def power_spectrum(self, k_array: Sequence[float] | np.ndarray, z: float) -> np.ndarray:
        arr = np.asarray(k_array, dtype=float)
        amplitude = self._amplitude / (1.0 + float(z))
        return amplitude * np.power(arr, self._slope)

    def pk_config(self) -> dict[str, str]:
        return {"k_units": "1/Mpc", "P_units": "Mpc^3"}


class _ToyPowerLawModel:
    def __init__(self, *, amplitude: float = 1.0, slope: float = 1.0, H0: float = 70.0) -> None:
        self.parameters = {"H0": float(H0)}
        self.matter = _ToyPowerLawMatter(amplitude=amplitude, slope=slope)


def _nearest_valid_index(k_values: np.ndarray, mask: np.ndarray, target: float) -> int:
    valid_indices = np.where(mask)[0]
    if valid_indices.size == 0:
        raise AssertionError("No valid k samples available.")
    return valid_indices[np.argmin(np.abs(k_values[valid_indices] - target))]


def test_pk_spectrum_prediction_matches_power_law() -> None:
    model = _ToyPowerLawModel(amplitude=1.3, slope=1.0)
    result = run_prediction_for_model("pk-spectrum", model, {"n_k": 128})
    assert result.status == "success"
    payload = result.results
    assert payload["name"] == "pk-spectrum"
    k_values = np.asarray(payload["k"], dtype=float)
    mask_valid = np.asarray(payload["mask_valid"], dtype=bool)
    assert k_values.size == 128
    assert mask_valid.size == k_values.size
    assert mask_valid.any()
    summary = payload["summary"]
    assert summary["P0p1_z0"] is not None
    assert summary["P0p2_z0"] is not None
    assert summary["P0p1_z1"] is not None

    idx_01 = _nearest_valid_index(k_values, mask_valid, 0.1)
    expected_01 = model.matter.power_spectrum(np.asarray([k_values[idx_01]]), 0.0)[0]
    assert math.isclose(summary["P0p1_z0"], expected_01, rel_tol=5e-3)

    idx_02 = _nearest_valid_index(k_values, mask_valid, 0.2)
    expected_02 = model.matter.power_spectrum(np.asarray([k_values[idx_02]]), 0.0)[0]
    assert math.isclose(summary["P0p2_z0"], expected_02, rel_tol=5e-3)

    idx_z1 = _nearest_valid_index(k_values, mask_valid, 0.1)
    expected_z1 = model.matter.power_spectrum(np.asarray([k_values[idx_z1]]), 1.0)[0]
    assert math.isclose(summary["P0p1_z1"], expected_z1, rel_tol=5e-3)

    sigma8_like = summary["sigma8_like"]
    assert sigma8_like is not None and math.isfinite(sigma8_like)
    json.dumps(payload)


def test_pk_spectrum_sigma8_scales_with_amplitude() -> None:
    low = _ToyPowerLawModel(amplitude=1.0)
    high = _ToyPowerLawModel(amplitude=2.0)
    low_result = run_prediction_for_model("pk-spectrum", low, {"n_k": 120})
    high_result = run_prediction_for_model("pk-spectrum", high, {"n_k": 120})
    low_sigma = low_result.results["summary"]["sigma8_like"]
    high_sigma = high_result.results["summary"]["sigma8_like"]
    assert low_sigma is not None and high_sigma is not None
    assert math.isclose(high_sigma / low_sigma, 2.0, rel_tol=1e-4)


class _ToyWeakLensingModel:
    def __init__(self, *, H0: float = 70.0) -> None:
        self.parameters = {"H0": float(H0)}
        self._H0 = float(H0)

    def Hubble(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        values = np.full(arr.shape, self._H0, dtype=float)
        return float(values[0]) if np.isscalar(z) else values

    def comoving_distance(self, z: float | Sequence[float]) -> float | np.ndarray:
        arr = np.atleast_1d(np.asarray(z, dtype=float))
        return float(arr[0]) if np.isscalar(z) else arr


def test_wl_kernel_prediction_behaviour() -> None:
    model = _ToyWeakLensingModel()
    config = {"zmin": 0.0, "zmax": 3.0, "points": 200}
    result = run_prediction_for_model("wl-kernel", model, config)
    assert result.status == "success"

    payload = result.results
    assert payload["name"] == "wl-kernel"
    z_arr = np.asarray(payload["z"], dtype=float)
    chi_vals = np.asarray(payload["chi"], dtype=float)
    n_z = np.asarray(payload["n_z"], dtype=float)
    W_raw = np.asarray(payload["W_raw"], dtype=float)
    W_norm = np.asarray(payload["W_norm"], dtype=float)
    mask_valid = np.asarray(payload["mask_valid"], dtype=bool)

    assert z_arr.shape == chi_vals.shape == n_z.shape == W_raw.shape == W_norm.shape == mask_valid.shape
    assert np.all(np.isfinite(z_arr))
    assert np.all(np.isfinite(chi_vals))
    assert np.all(n_z >= 0.0)

    summary = payload["summary"]
    assert summary["z_peak"] is not None
    assert 0.1 < float(summary["z_peak"]) < 1.5
    assert summary["z_median"] is not None
    assert 0.1 < float(summary["z_median"]) < 1.5
    assert summary["W_peak_value"] is not None
    assert summary["W_peak_value"] >= 0.0

    valid_raw = W_raw[mask_valid]
    assert np.all(np.isfinite(valid_raw))
    assert np.all(valid_raw >= 0.0)
    assert np.all(np.isfinite(W_norm[mask_valid]))
    assert np.count_nonzero(mask_valid) >= 10

    assert result.metadata["summary"]["z_peak"] == summary["z_peak"]
    assert payload["meta"]["n_z_model"] == "default"

    json.dumps(payload)


def test_wl_kernel_prediction_source_config_and_raw_mode() -> None:
    model = _ToyWeakLensingModel()
    config = {
        "zmin": 0.0,
        "zmax": 2.5,
        "points": 150,
        "normalize": False,
        "source_distribution": {
            "type": "euclid_like",
            "parameters": {"z0": 0.9},
        },
    }
    result = run_prediction_for_model("wl-kernel", model, config)
    payload = result.results
    mask_valid = np.asarray(payload["mask_valid"], dtype=bool)
    W_raw = np.asarray(payload["W_raw"], dtype=float)
    W_norm = np.asarray(payload["W_norm"], dtype=float)

    assert payload["meta"]["n_z_model"] == "euclid_like"
    assert mask_valid.any()
    assert np.array_equal(W_raw[mask_valid], W_norm[mask_valid])
