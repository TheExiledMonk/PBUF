#!/usr/bin/env python
import numpy as np

from cosmos2.models.model_factory import create_model
from cosmos2.science_runner.runner import _load_pbuf_lut
from pbuf_cmb_reference import compute_pbuf_cmb


def main():
    params = dict(
        H0=73.8,
        Omega_m0=0.26844,
        Omega_b0=0.0235,
        Omega_k0=0.0,
        Omega_r0=9.0e-5,
        alpha=0.022042,
        Rmax=7.87e6,
        k_sat=0.933891,
        thermal_mode="off",
    )

    lut = _load_pbuf_lut()
    model = create_model("pbuf", lut=lut, **params)
    cmb_out = model.cmb(None)
    R_cos = float(cmb_out.R)
    lA_cos = float(cmb_out.l_A)
    th_cos = float(cmb_out.theta_star)

    R_ref, lA_ref, th_ref = compute_pbuf_cmb(params)

    print("Cosmos2:   R, lA, theta =", R_cos, lA_cos, th_cos)
    print("Reference: R, lA, theta =", R_ref, lA_ref, th_ref)

    for name, a, b in [
        ("R", R_cos, R_ref),
        ("lA", lA_cos, lA_ref),
        ("theta", th_cos, th_ref),
    ]:
        diff = abs(a - b)
        rel = diff / max(abs(b), 1e-6)
        print(f"{name}: diff={diff}, rel={rel}")
        if rel > 1.0e-3:
            print("WARNING:", name, "mismatch above tolerance")


if __name__ == "__main__":
    main()
