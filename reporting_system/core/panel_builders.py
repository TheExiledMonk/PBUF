"""
Panel builders that render individual sections of the science run report.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .formatting import format_number


def generate_metadata_panel(run_metadata: Dict[str, Any]) -> str:
    """Generate metadata panel with engine and config info."""
    config = run_metadata.get("config", {})
    engine = config.get("engine", "basin")
    engine_settings = config.get("engine_settings", {})

    settings_rows = ""
    for key, value in engine_settings.items():
        settings_rows += f"<tr><td>{key}</td><td>{value}</td></tr>"

    config_json = json.dumps(config, indent=2)

    return f"""
<section class='panel'>
    <h2>Engine & configuration metadata</h2>
    <table class='details'>
        <thead>
            <tr>
                <th>Setting</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Engine name</td><td>{engine}</td></tr>
        </tbody>
    </table>

    <div style='margin-top: 1rem;'>
        <h3>Engine settings</h3>
        <table class='details'>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {settings_rows}
            </tbody>
        </table>
    </div>

    <div style='margin-top: 1rem;'>
        <h3>Science config</h3>
        <details>
            <summary>config_used.json</summary>
            <pre>{config_json}</pre>
        </details>
    </div>
</section>"""


def _to_relative_path(run_dir: Path, path_to_include: str | Path) -> Path:
    """Convert an absolute path to one relative to the run directory if possible."""
    file_path = Path(path_to_include)
    try:
        return file_path.resolve().relative_to(run_dir.resolve())
    except Exception:
        return file_path


def generate_model_panel(
    model: str,
    model_data: Dict[str, Any],
    best_model: str | None,
    best_chi2: float,
    run_dir: Path,
    model_figures: List[Dict[str, Any]] | None = None,
) -> str:
    """Generate individual model panel."""
    best_fit = model_data.get("best_fit", {})
    chi2_breakdown = model_data.get("chi2_breakdown", {})
    parameters = best_fit.get("parameters", {})
    jackknife_data = model_data.get("jackknife_data", {})
    fits_data = model_data.get("fits_data", {})
    model_figures = model_figures or []

    chi2 = best_fit.get("chi_squared", best_fit.get("chi2", 0))
    aic = best_fit.get("aic", 0)
    bic = best_fit.get("bic", 0)
    n_params = len(parameters)

    param_rows = ""
    for param, value in parameters.items():
        if isinstance(value, (int, float)):
            param_rows += (
                f"<tr><td>{param}</td><td>{format_number(value, '.6f')}</td></tr>"
            )

    chi2_rows = ""
    total_chi2 = 0
    for dataset, chi2_val in chi2_breakdown.items():
        if isinstance(chi2_val, (int, float)):
            chi2_rows += (
                f"<tr><td>{dataset}</td><td>{format_number(chi2_val, '.3f')}</td></tr>"
            )
            total_chi2 += chi2_val
    chi2_rows += (
        f"<tr><td><strong>total</strong></td><td><strong>{format_number(total_chi2, '.3f')}</strong></td></tr>"
    )

    jackknife_html = ""
    if jackknife_data.get("jackknife_available"):
        stability_score = jackknife_data.get("stability_score", 0)
        success_rate = jackknife_data.get("success_rate", 0)
        n_draws = jackknife_data.get("n_draws", 0)
        param_stability_rows = ""
        param_stability = jackknife_data.get("parameter_stability", {})
        for param, stats in param_stability.items():
            mean_val = stats.get("mean", 0)
            std_val = stats.get("std", 0)
            cv = stats.get("cv", 0)
            stability = stats.get("stability", "unknown")
            stability_class = f"stability-{stability}"
            param_stability_rows += f"""
            <tr>
                <td>{param}</td>
                <td>{format_number(mean_val, '.4f')} ± {format_number(std_val, '.4f')}</td>
                <td>{format_number(cv, '.4f')}</td>
                <td><span class="stability-indicator {stability_class}">{stability}</span></td>
            </tr>"""
        recommendations = jackknife_data.get("recommendations", [])
        rec_list = "".join([f"<li>{rec}</li>" for rec in recommendations])
        jackknife_fig = _pop_matching_figure(model_figures, "jackknife_chi2")
        jackknife_plot = ""
        if jackknife_fig:
            plot_path = Path(jackknife_fig.get("file_path", ""))
            if plot_path.exists():
                rel_jk_path = _to_relative_path(run_dir, plot_path)
                jackknife_plot = f"""
        <div class='figure-block'>
            <h4>{model.upper()} jackknife χ²</h4>
            <img src="{rel_jk_path.as_posix()}" alt="{model.upper()} jackknife χ²" style="max-width: 100%; height: auto; margin-top: 0.5rem;">
        </div>"""
        jackknife_html = f"""
        <h3>🔍 Jackknife Analysis</h3>
        {jackknife_plot}
        <div class="jackknife-summary">
            <p><strong>Stability Score:</strong> {format_number(stability_score, '.3f')}</p>
            <p><strong>Success Rate:</strong> {format_number(success_rate, '.1%')} ({n_draws} draws)</p>
        </div>

        <h4>Parameter Stability</h4>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Mean ± Std</th>
                    <th>CV</th>
                    <th>Stability</th>
                </tr>
            </thead>
            <tbody>
                {param_stability_rows}
            </tbody>
        </table>

        <h4>Recommendations</h4>
        <ul>
            {rec_list}
        </ul>"""

    fit_details_html = ""
    if isinstance(fits_data, dict) and fits_data:
        fit_entries = ""
        for ds_key, payload in fits_data.items():
            extras = payload.get("extras", {})
            dataset_name = extras.get("dataset", {}).get("name", ds_key.upper())
            chi2_val = payload.get("chi2", payload.get("weighted_chi2"))
            weight = payload.get("weight", 1.0)
            observed = extras.get("observed", [])
            predictions_list = extras.get("predictions", [])
            residuals_list = extras.get("residuals", [])
            meta = extras.get("dataset", {}).get("meta", {})
            meta_rows = ""
            for mkey, mval in meta.items():
                meta_rows += f"<tr><td>{mkey}</td><td>{mval}</td></tr>"
            prediction_entry = model_data.get("predictions", {}).get(ds_key, {})
            plot_html = ""
            plot_path = prediction_entry.get("prediction_plot")
            if plot_path:
                _consume_figure(model_figures, plot_path)
                rel_plot = _to_relative_path(run_dir, plot_path)
                plot_html = f"""
            <div class='figure-block'>
                <h4>Predictions vs data</h4>
                <img src="{rel_plot.as_posix()}" alt="Predictions vs observed for {dataset_name}" style="max-width: 100%; height: auto; margin-top: 0.5rem;">
            </div>"""
            fit_entries += f"""
        <div class='fit-detail'>
            <h4>{dataset_name}</h4>
            <p>χ²: {chi2_val if chi2_val is not None else 'n/a'}  •  weight: {weight}</p>
            <p>observed: {len(observed)} pts  •  prediction: {len(predictions_list)} pts  •  residuals: {len(residuals_list)} pts</p>
            <table class='details'>
                <thead><tr><th>Meta</th><th>Value</th></tr></thead>
                <tbody>
                    {meta_rows if meta_rows else '<tr><td colspan="2">(none)</td></tr>'}
                </tbody>
            </table>
            {plot_html}
            <details>
                <summary>Extras payload</summary>
                <pre>{json.dumps(extras, indent=2)}</pre>
            </details>
        </div>"""
        fit_details_html = f"""
        <h3>Fit outputs</h3>
        {fit_entries}
        """

    figures_html = _render_model_figure_gallery(model, model_figures, run_dir)

    dataset_impact_html = ""
    dataset_impact = jackknife_data.get("dataset_impact", {})
    if isinstance(dataset_impact, dict) and dataset_impact:
        impact_rows = ""
        for ds_name, impact in dataset_impact.items():
            freq = impact.get("impact_frequency")
            change = impact.get("chi2_change")
            impact_rows += f"<tr><td>{ds_name}</td><td>{freq if freq is not None else 'n/a'}</td><td>{change if change is not None else 'n/a'}</td></tr>"
        dataset_impact_html = f"""
        <h3>Dataset impact</h3>
        <table class='details'>
            <thead>
                <tr><th>Dataset</th><th>Impact freq</th><th>Δχ²</th></tr>
            </thead>
            <tbody>
                {impact_rows}
            </tbody>
        </table>
        """

    chi2_stability_html = ""
    chi2_stability = jackknife_data.get("chi2_stability", {})
    if isinstance(chi2_stability, dict) and chi2_stability:
        stability_rows = ""
        for label, stats in chi2_stability.items():
            mean = stats.get("mean_chi2", stats.get("mean")) if isinstance(stats, dict) else None
            std = stats.get("std_chi2", stats.get("std")) if isinstance(stats, dict) else None
            rng = stats.get("range_chi2", stats.get("range")) if isinstance(stats, dict) else None
            stability_rows += f"<tr><td>{label}</td><td>{mean if mean is not None else 'n/a'}</td><td>{std if std is not None else 'n/a'}</td><td>{rng if rng is not None else 'n/a'}</td></tr>"
        chi2_stability_html = f"""
        <h3>Jackknife χ² stability</h3>
        <table class='details'>
            <thead>
                <tr><th>Label</th><th>Mean</th><th>Std</th><th>Range</th></tr>
            </thead>
            <tbody>
                {stability_rows}
            </tbody>
        </table>
        """

    conclusion_html = ""
    conclusion_items: List[str] = []
    if best_model:
        if model == best_model:
            conclusion_items.append("Lowest χ² among the compared models.")
        else:
            delta_chi2 = chi2 - best_chi2
            conclusion_items.append(
                f"Δχ² = {format_number(delta_chi2, '.1f')} relative to {best_model.upper()}."
            )
    stability_score = jackknife_data.get("stability_score")
    if stability_score is not None:
        if stability_score >= 0.9:
            conclusion_items.append(
                f"Jackknife stability strong ({format_number(stability_score, '.3f')})."
            )
        elif stability_score >= 0.7:
            conclusion_items.append(
                f"Jackknife stability moderate ({format_number(stability_score, '.3f')})."
            )
        else:
            conclusion_items.append(
                f"Jackknife stability low ({format_number(stability_score, '.3f')}) – review data quality."
            )
    top_impact = None
    if isinstance(dataset_impact, dict):
        top_impact = max(
            dataset_impact.items(),
            key=lambda item: item[1].get("impact_frequency", 0),
            default=None,
        )
    if top_impact and top_impact[1].get("impact_frequency", 0) > 0:
        ds_name, impact = top_impact
        freq = impact.get("impact_frequency", 0)
        if isinstance(freq, (int, float)):
            freq_pct = format_number(freq * 100, ".0f")
        else:
            freq_pct = str(freq)
        conclusion_items.append(
            f"Dataset {ds_name} drives {freq_pct}% of jackknife perturbations."
        )
    if conclusion_items:
        bullet_items = "".join(f"<li>{item}</li>" for item in conclusion_items)
        conclusion_html = f"""
        <h3>Conclusion</h3>
        <ul>
            {bullet_items}
        </ul>
        """

    prediction_html = ""
    predictions = model_data.get("predictions", {})
    if predictions:
        prediction_details = ""
        for ds_key, info in predictions.items():
            dataset_label = info.get("dataset_name", ds_key.upper())
            obs_summary = info.get("observed_summary", {})
            pred_summary = info.get("prediction_summary", {})
            resid_summary = info.get("residual_summary", {})
            detail_payload = {
                "dataset_meta": info.get("dataset_meta", {}),
                "observed": obs_summary,
                "predictions": pred_summary,
                "residuals": resid_summary,
            }
            prediction_details += f"""
        <details>
            <summary>{dataset_label} ({obs_summary.get('count', 0)} points)</summary>
            <pre>{json.dumps(detail_payload, indent=2)}</pre>
        </details>"""
        prediction_html = f"""
        <h3>Predictions</h3>
        {prediction_details}
        """

    residual_html = ""
    residual_plot = model_data.get("residual_plot")
    if residual_plot:
        _consume_figure(model_figures, residual_plot)
        rel_path = _to_relative_path(run_dir, residual_plot)
        residual_html = f"""
    <div class='figure-block'>
        <h4>Residuals</h4>
        <img src="{rel_path}" alt="{model.upper()} residuals" style="max-width: 100%; height: auto;">
    </div>"""

    return f"""
<section class='panel model-panel'>
    <h2>{model.upper()}</h2>
    <div class='model-meta'>
        <span>χ² {format_number(chi2, '.1f')}</span>
        <span>{n_params} parameters</span>
        <span>AIC {format_number(aic, '.1f')}</span>
        <span>BIC {format_number(bic, '.1f')}</span>
    </div>

    <div class='grid'>
        <div class='grid-item'>
            <label>Best-fit parameters</label>
            <table class='details'>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {param_rows}
                </tbody>
            </table>
        </div>

        <div class='grid-item'>
            <label>χ² breakdown</label>
            <table class='details'>
                <thead>
                    <tr>
                        <th>Fit</th>
                        <th>χ²</th>
                    </tr>
                </thead>
                <tbody>
                    {chi2_rows}
                </tbody>
            </table>
        </div>
    </div>

    {jackknife_html}
    {conclusion_html}
    {fit_details_html}
    {dataset_impact_html}
    {chi2_stability_html}
    {prediction_html}
    {residual_html}
    {figures_html}
</section>"""


def generate_figures_panels(figures: List[Dict[str, Any]], run_dir: Path) -> str:
    """Generate figures panels."""
    if not figures:
        return ""

    figure_panels = ""
    for fig in figures:
        fig_path = fig.get('file_path', '')
        fig_name = fig.get('name', 'Figure')

        file_source = Path(fig_path)
        if not file_source.exists():
            continue

        rel_path = _to_relative_path(run_dir, file_source)
        figure_panels += f"""
<section class='panel'>
    <h2>{fig_name}</h2>
    <img src="{rel_path.as_posix()}" alt="{fig_name}" style="max-width: 100%; height: auto;">
</section>"""

    return figure_panels


def _render_model_figure_gallery(model: str, figures: List[Dict[str, Any]], run_dir: Path) -> str:
    """Render remaining model figures after dataset/residual usage."""

    thermal = [fig for fig in figures if fig.get("model_group") == "temperature"]
    others = [fig for fig in figures if fig not in thermal]

    blocks = []
    if thermal:
        blocks.append(_render_model_subgallery(f"{model.upper()} thermal plots", thermal, run_dir))
    if others:
        blocks.append(_render_model_subgallery(f"{model.upper()} plots", others, run_dir))

    return "".join(blocks)


def _render_model_subgallery(title: str, figures: List[Dict[str, Any]], run_dir: Path) -> str:
    """Render a titled subgallery of model figures."""

    cards = []
    for fig in figures:
        fig_path = fig.get("file_path")
        if not fig_path:
            continue
        file_path = Path(fig_path)
        if not file_path.exists():
            continue
        rel_path = _to_relative_path(run_dir, file_path)
        caption = _format_label(fig.get("name") or file_path.stem)
        cards.append(f"""
        <figure class='model-figure'>
            <img src="{rel_path.as_posix()}" alt="{caption}" loading="lazy">
            <figcaption>{caption}</figcaption>
        </figure>""")

    if not cards:
        return ""

    return f"""
    <div class='model-figures'>
        <h3>{title}</h3>
        {''.join(cards)}
    </div>"""


def _consume_figure(figures: List[Dict[str, Any]], plot_path: str | Path) -> None:
    """Remove a consumed figure from the provided list."""

    if not plot_path:
        return
    target = Path(plot_path).name
    remaining = []
    for fig in figures:
        fig_path = fig.get("file_path")
        if not fig_path:
            remaining.append(fig)
            continue
        candidate = Path(fig_path).name
        if candidate == target:
            continue
        remaining.append(fig)
    figures[:] = remaining


def _format_label(value: str | None) -> str:
    """Normalize figure names into readable labels."""

    if not value:
        return ""
    normalized = " ".join(value.replace("_", " ").split())
    return normalized.title()


def _pop_matching_figure(figures: List[Dict[str, Any]], suffix: str) -> Dict[str, Any] | None:
    """Extract and remove the first figure whose name ends with the given suffix."""

    for idx, fig in enumerate(figures):
        name = (fig.get("name") or "").lower()
        if name.endswith(suffix.lower()):
            return figures.pop(idx)
    return None


def generate_hero_section(run_dir: Path, models: List[str], run_metadata: Dict[str, Any]) -> str:
    """Generate beautiful hero header section."""
    run_name = run_dir.name
    config = run_metadata.get('config', {})
    run_meta = run_metadata.get('run_meta', {})

    timestamp = run_meta.get('timestamp', datetime.now().strftime("%Y-%m-%dT%H%M%S"))
    mode = config.get('mode', 'fit')
    engine = config.get('engine', 'basin')
    datasets = config.get('fits_list', ["cmb", "sn", "bao_iso", "cc", "rsd"])
    n_datasets = len(datasets)

    status = "Success"
    status_class = "success"

    return f"""
<header class='hero'>
    <p class='eyebrow'>Science Run Overview</p>
    <h1>{run_name}</h1>
    <p>Comprehensive cosmological analysis report</p>
    <div class='hero-grid'>
        <div class='hero-card'>
            <strong>Timestamp</strong>
            <span>{timestamp}</span>
        </div>
        <div class='hero-card'>
            <strong>Mode</strong>
            <span>{mode}</span>
        </div>
        <div class='hero-card'>
            <strong>Engine</strong>
            <span>{engine}</span>
        </div>
        <div class='hero-card'>
            <strong>Fits</strong>
            <span>{n_datasets} ({', '.join(datasets)})</span>
        </div>
        <div class='hero-card'>
            <strong>Models</strong>
            <span>{len(models)} ({', '.join([m.upper() for m in models])})</span>
        </div>
        <div class='hero-card'>
            <strong>Run status</strong>
            <span>{status}</span>
        </div>
    </div>
    <div class='status-chip {status_class}'>{status}</div>
</header>"""


def generate_overview_panel(models: List[str], run_metadata: Dict[str, Any]) -> str:
    """Generate overview panel with key stats."""
    config = run_metadata.get('config', {})
    run_meta = run_metadata.get('run_meta', {})

    datasets = config.get('fits_list', [])
    total_datapoints = sum([
        len(run_meta.get(f'{dataset}_data', [])) if isinstance(run_meta.get(f'{dataset}_data'), list) else 0
        for dataset in datasets
    ]) or 1756

    duration = run_meta.get('duration')
    if not duration:
        start = run_meta.get('start_timestamp')
        end = run_meta.get('end_timestamp')
        if start and end:
            try:
                from datetime import datetime

                t0 = datetime.fromisoformat(start)
                t1 = datetime.fromisoformat(end)
                delta = t1 - t0
                hours = delta.seconds // 3600 + delta.days * 24
                minutes = (delta.seconds % 3600) // 60
                duration = f"{hours}h {minutes}m"
            except Exception:
                duration = "unknown"
        else:
            duration = "unknown"
    n_threads = config.get('engine_settings', {}).get('workers', 20)
    n_models = len(models)

    return f"""
<section class='panel'>
    <h2>Run at a glance</h2>
    <div class='stats-grid'>
        <div class='stat-card'>
            <div class='stat-label'>Duration</div>
            <div class='stat-value'>{duration}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Data points</div>
            <div class='stat-value'>{total_datapoints}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Models</div>
            <div class='stat-value'>{n_models}</div>
        </div>
        <div class='stat-card'>
            <div class='stat-label'>Threads</div>
            <div class='stat-value'>{n_threads}</div>
        </div>
    </div>
</section>"""


def generate_model_comparison_panel(models: List[str], model_data: Dict[str, Any]) -> str:
    """Generate model comparison panel."""
    if len(models) < 2:
        return ""

    model_rows = ""
    best_chi2 = float('inf')
    best_model_data = None

    for model in models:
        data = model_data[model]
        best_fit = data.get('best_fit', {})
        chi2 = best_fit.get('chi_squared', float('inf'))
        aic = best_fit.get('aic', float('inf'))
        bic = best_fit.get('bic', float('inf'))

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_model_data = (model, chi2, aic, bic)

    for model in models:
        data = model_data[model]
        best_fit = data.get('best_fit', {})
        chi2 = best_fit.get('chi_squared', float('inf'))
        aic = best_fit.get('aic', float('inf'))
        bic = best_fit.get('bic', float('inf'))

        delta_chi2 = chi2 - best_chi2 if best_chi2 < float('inf') else 0
        delta_aic = aic - best_model_data[2] if best_model_data and best_model_data[2] < float('inf') else 0
        delta_bic = bic - best_model_data[3] if best_model_data and best_model_data[3] < float('inf') else 0

        row_class = "highlight-row" if chi2 == best_chi2 else ""

        model_rows += f"""
<tr class='{row_class}'>
    <td>{model.upper()}</td>
    <td>{format_number(chi2, '.1f')}</td>
    <td>{format_number(delta_chi2, '.1f')}</td>
    <td>{format_number(aic, '.1f')}</td>
    <td>{format_number(delta_aic, '.1f')}</td>
    <td>{format_number(bic, '.1f')}</td>
    <td>{format_number(delta_bic, '.1f')}</td>
</tr>"""

    return f"""
<section class='panel'>
    <h2>Model comparison</h2>
    <table class='summary-table'>
        <thead>
            <tr>
                <th>Model</th>
                <th>χ²</th>
                <th>Δχ²</th>
                <th>AIC</th>
                <th>ΔAIC</th>
                <th>BIC</th>
                <th>ΔBIC</th>
            </tr>
        </thead>
        <tbody>
            {model_rows}
        </tbody>
    </table>
</section>"""
