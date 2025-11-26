"""Formatting helpers for CMB fit results."""

from __future__ import annotations

from cosmos.interfaces import CMBOutput


def _format_section(title: str, rows: list[tuple[str, str, str]]) -> list[str]:
    if not rows:
        return []

    PARAM_WIDTH = 18
    VALUE_WIDTH = 14
    lines = [
        title,
        "-" * len(title),
        f"{'Parameter':<{PARAM_WIDTH}} {'Value':>{VALUE_WIDTH}} Unit",
        "-" * (PARAM_WIDTH + VALUE_WIDTH + 7),
    ]

    for label, value, unit in rows:
        lines.append(f"{label:<{PARAM_WIDTH}} {value:>{VALUE_WIDTH}} {unit}")

    return lines


def render_report(result: dict) -> str:
    output: CMBOutput = result["output"]
    chi2 = result["chi2"]
    model_name = result["model_name"]

    title = f"CMB Fit Report [{model_name.upper()}]"
    lines = [title, "=" * len(title)]

    summary_rows = [("chi^2", f"{chi2:0.3f}", "")]
    derived_rows = [
        ("R", f"{output.R:0.5f}", ""),
        ("l_A", f"{output.l_A:0.5f}", ""),
        ("Omega_b h^2", f"{output.Omega_b_h2:0.5f}", ""),
        ("theta_star", f"{output.theta_star:0.5e}", ""),
        ("z_star", f"{output.z_star:0.2f}", ""),
        ("D_M", f"{output.D_M_Mpc:0.2f}", "Mpc"),
        ("D_A", f"{output.D_A_Mpc:0.2f}", "Mpc"),
        ("r_s", f"{output.r_s_Mpc:0.2f}", "Mpc"),
    ]

    lines.extend(["", *_format_section("Fit Overview", summary_rows)])
    lines.extend(["", *_format_section("Derived Parameters", derived_rows)])

    extras = output.extras or {}
    if extras:
        lines.extend(
            [
                "",
                "Extras",
                "-" * 6,
                *[f"{key}: {value}" for key, value in extras.items()],
            ]
        )

    return "\n".join(lines)
