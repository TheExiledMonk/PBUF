import math

import quantum.api as api
from quantum.api import ThermalSpec, run_quantum_engine
from quantum.thermal.fitter import MicrophysicsInputs, ThermalFitResult, derive_thermal_params


def test_run_quantum_engine_derives_params(monkeypatch):
    raw = {
        "eps0": 1.05,
        "alpha_QM": 0.021,
        "derived_parameters": {
            "regulator": "hard_cutoff",
            "field_set": "SM_full",
            "f_cut": 0.675,
            "f_coup": 0.12,
            "mixing_strength": 0.3,
        },
        "run_metadata": {},
    }

    spec = ThermalSpec(
        temperature_points=64,
        dense_points=0,
        t_min=2.7255,
        t_max=1.0e6,
        table_version=12,
        method_version=12,
        fit_samples=64,
        fit_min=1.0e3,
        fit_max=1.0e8,
        fit_points=32,
    )

    def _fake_legacy(_override=None):
        return raw

    micro_inputs = MicrophysicsInputs(
        eps0_today=raw["eps0"],
        alpha_qm=raw["alpha_QM"],
        regulator=raw["derived_parameters"]["regulator"],
        field_content=raw["derived_parameters"]["field_set"],
        f_coup=raw["derived_parameters"]["f_coup"],
        mixing_strength=raw["derived_parameters"]["mixing_strength"],
    )
    expected_fit = derive_thermal_params(
        micro_inputs,
        t_min=spec.t_min,
        t_max=spec.t_max,
        samples=spec.fit_samples,
        fit_min=spec.fit_min,
        fit_max=spec.fit_max,
        fit_points=spec.fit_points,
    )

    monkeypatch.setattr(api, "_legacy_engine", _fake_legacy)
    monkeypatch.setattr(api, "_load_spec", lambda: spec)

    result = run_quantum_engine()

    assert math.isclose(result["beta"], expected_fit.beta, rel_tol=1e-3, abs_tol=1e-6)
    assert math.isclose(result["T_star"], expected_fit.t_star, rel_tol=1e-3, abs_tol=1e-6)
    assert math.isclose(result["power_index"], expected_fit.power, rel_tol=1e-3, abs_tol=1e-6)
    assert result["regulator"] == "hard_cutoff"
    assert result["field_content"] == "SM_full"
