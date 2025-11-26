"""Quick helper to compare H(a) between LCDM and PBUF at a few redshifts."""

from __future__ import annotations

from cosmos2.science_runner.runner import _load_pbuf_lut
from cosmos2.models.model_factory import create_model


def main() -> None:
    z_samples = [0.1, 0.3, 0.5, 1.0, 2.0]
    common_params = {
        "H0": 67.4,
        "Omega_m0": 0.315,
        "Omega_b0": 0.049,
    }
    lut = _load_pbuf_lut()

    lcdm = create_model("lcdm", **common_params)
    pbuf_off = create_model("pbuf", lut=lut, thermal_mode="off", **common_params)
    pbuf_power = create_model("pbuf", lut=lut, thermal_mode="power", **common_params)

    print("z  H_lcdm  H_pbuf_off  H_pbuf_power  (km/s/Mpc)")
    for z in z_samples:
        h_lcdm = float(lcdm.Hubble(z))
        h_off = float(pbuf_off.Hubble(z))
        h_power = float(pbuf_power.Hubble(z))
        print(f"{z:3.1f} {h_lcdm:10.4f} {h_off:12.4f} {h_power:14.4f}")


if __name__ == "__main__":
    main()
