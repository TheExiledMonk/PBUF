import math

from cosmos.physics.sigma8 import E_LCDM, sigma8_from_primordial


def test_sigma8_planck_like_lcdm():
    As = 2.1e-9
    ns = 0.965
    H0 = 67.36
    h = H0 / 100.0
    Om0 = 0.315
    Or0 = 9.2e-5
    Ok0 = 0.0
    Obh2 = 0.02237

    E_ref = lambda a: E_LCDM(a, H0=H0, Om0=Om0, Or0=Or0, Ok0=Ok0)

    s8 = sigma8_from_primordial(
        As=As,
        ns=ns,
        h=h,
        Om0=Om0,
        Obh2=Obh2,
        E_of_a=E_ref,
        calibrate_against=dict(
            As=As,
            ns=ns,
            h=h,
            Om0=Om0,
            Obh2=Obh2,
            E_of_a=E_ref,
            sigma8_ref=0.811,
        ),
        calibration_key="planck18_lcdm",
    )
    assert abs(s8 - 0.811) < 1e-3


def test_sigma8_relative_shift_with_growth_change():
    As = 2.1e-9
    ns = 0.965
    H0_a = 67.36
    H0_b = 73.0
    h_a, h_b = H0_a / 100.0, H0_b / 100.0
    Om0 = 0.315
    Or0 = 9.2e-5
    Ok0 = 0.0
    Obh2 = 0.02237

    E_a = lambda a: E_LCDM(a, H0=H0_a, Om0=Om0, Or0=Or0, Ok0=Ok0)
    E_b = lambda a: E_LCDM(a, H0=H0_b, Om0=Om0, Or0=Or0, Ok0=Ok0)

    s8_a = sigma8_from_primordial(
        As=As,
        ns=ns,
        h=h_a,
        Om0=Om0,
        Obh2=Obh2,
        E_of_a=E_a,
        calibrate_against=dict(
            As=As,
            ns=ns,
            h=h_a,
            Om0=Om0,
            Obh2=Obh2,
            E_of_a=E_a,
            sigma8_ref=0.811,
        ),
        calibration_key="planck18_lcdm",
    )
    s8_b = sigma8_from_primordial(
        As=As,
        ns=ns,
        h=h_b,
        Om0=Om0,
        Obh2=Obh2,
        E_of_a=E_b,
        calibrate_against=dict(
            As=As,
            ns=ns,
            h=h_a,
            Om0=Om0,
            Obh2=Obh2,
            E_of_a=E_a,
            sigma8_ref=0.811,
        ),
        calibration_key="planck18_lcdm",
    )
    assert math.isfinite(s8_b) and abs(s8_b - s8_a) > 1e-3
