"""Eclipse and rotational-modulation diagnostics for saved TESS light curves."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
from astropy.timeseries import BoxLeastSquares, LombScargle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


R_SUN_KM = 695700.0
DAY_S = 86400.0


def _finite_float(value, default=np.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _resolve_path(value, output_root):
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    candidates = [path]
    if not path.is_absolute():
        output_root = Path(output_root)
        candidates.extend((output_root / path, output_root.parent / path))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def select_lightcurve_files(manifest, output_root):
    """Select one existing pipeline light curve per Frame.

    SPOC is preferred, followed by the product with the largest cadence count.
    Keeping one product per Frame bounds both runtime and notebook output.
    """
    required = {"frame_id", "status", "fits_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    selected = manifest.loc[manifest["status"].eq("saved")].copy()
    selected["resolved_fits_path"] = selected["fits_path"].map(
        lambda value: _resolve_path(value, output_root)
    )
    selected = selected.loc[selected["resolved_fits_path"].notna()].copy()
    if selected.empty:
        return selected
    author = selected.get("author", pd.Series("", index=selected.index)).fillna("")
    selected["_spoc_priority"] = author.astype(str).str.upper().eq("SPOC")
    selected["_n_cadences"] = pd.to_numeric(
        selected.get("n_cadences", np.nan), errors="coerce"
    ).fillna(-1)
    selected = selected.sort_values(
        ["frame_id", "_spoc_priority", "_n_cadences"],
        ascending=[True, False, False],
    ).drop_duplicates("frame_id", keep="first")
    return selected.drop(columns=["_spoc_priority", "_n_cadences"])


def rebuild_diagnostic_figures(
    manifest, output_root, *, dpi=160, overwrite=False, update_manifest=True
):
    """Rebuild one diagnostic PNG per Frame from local manifest products only.

    When ``update_manifest`` is true, every saved row for the rebuilt Frame is
    pointed at its single stable ``diagnostic.png`` and the CSV is checkpointed.
    """
    import lightkurve as lk
    from tess_diagnostics import save_diagnostic_figure

    output_root = Path(output_root)
    selected = select_lightcurve_files(manifest, output_root)
    rows = []
    for _, item in tqdm(
        selected.iterrows(), total=len(selected), desc="Diagnostic figures", unit="frame"
    ):
        frame_id = str(item["frame_id"])
        figure_path = output_root / frame_id / "diagnostic.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        base = {"frame_id": frame_id, "diagnostic_figure_path": str(figure_path)}
        if figure_path.is_file() and not overwrite:
            rows.append({**base, "diagnostic_status": "existing"})
            continue
        try:
            cleaned = lk.read(item["resolved_fits_path"]).remove_nans().normalize()
            tpf_path = _resolve_path(item.get("tpf_path"), output_root)
            tpf = lk.read(tpf_path) if tpf_path is not None else None
            sector = item.get("sector")
            title = (
                f"{frame_id} | {item.get('object_name')} | "
                f"TESS sector {sector} | {item.get('author')}"
            )
            best_period = save_diagnostic_figure(
                cleaned, tpf, figure_path, title, dpi=dpi
            )
            rows.append({
                **base, "diagnostic_status": "saved",
                "best_lombscargle_period_day": best_period,
            })
        except Exception as exc:
            rows.append({**base, "diagnostic_status": "error", "error": repr(exc)})
    updates = pd.DataFrame.from_records(rows)
    if update_manifest and not updates.empty:
        successful = updates.loc[
            updates["diagnostic_status"].isin(("saved", "existing"))
        ].copy()
        figure_by_frame = successful.set_index("frame_id")["diagnostic_figure_path"]
        selected_rows = manifest["status"].eq("saved") & manifest["frame_id"].astype(str).isin(
            figure_by_frame.index
        )
        manifest.loc[selected_rows, "diagnostic_figure_path"] = (
            manifest.loc[selected_rows, "frame_id"].astype(str).map(figure_by_frame)
        )
        if "best_lombscargle_period_day" in successful:
            period_by_frame = successful.dropna(
                subset=["best_lombscargle_period_day"]
            ).set_index("frame_id")["best_lombscargle_period_day"]
            period_rows = selected_rows & manifest["frame_id"].astype(str).isin(
                period_by_frame.index
            )
            manifest.loc[period_rows, "best_lombscargle_period_day"] = (
                manifest.loc[period_rows, "frame_id"].astype(str).map(period_by_frame)
            )
        manifest_path = output_root / "manifest.csv"
        temporary = manifest_path.with_suffix(".csv.tmp")
        manifest.to_csv(temporary, index=False)
        temporary.replace(manifest_path)
    return updates


def _load_lightcurve_arrays(path):
    import lightkurve as lk

    lightcurve = lk.read(path).remove_nans().normalize()
    time = np.asarray(lightcurve.time.value, dtype=float)
    flux = np.asarray(getattr(lightcurve.flux, "value", lightcurve.flux), dtype=float)
    flux_err = np.asarray(
        getattr(lightcurve.flux_err, "value", lightcurve.flux_err), dtype=float
    )
    valid = np.isfinite(time) & np.isfinite(flux)
    time, flux, flux_err = time[valid], flux[valid], flux_err[valid]
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    unique = np.concatenate(([True], np.diff(time) > 0))
    time, flux, flux_err = time[unique], flux[unique], flux_err[unique]
    if time.size < 100:
        raise ValueError(f"Only {time.size} finite, unique cadences in {path}")
    flux /= np.nanmedian(flux)
    return time - time[0], flux, flux_err


def _robust_sigma(values):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    return 1.4826 * np.nanmedian(np.abs(values - median))


def _phase_metrics(time, flux, period, n_bins=24):
    phase = np.mod(time, period) / period
    bin_index = np.minimum((phase * n_bins).astype(int), n_bins - 1)
    medians = np.full(n_bins, np.nan)
    for index in range(n_bins):
        values = flux[bin_index == index]
        if values.size >= 5:
            medians[index] = np.nanmedian(values)
    available = np.isfinite(medians)
    if available.sum() < n_bins // 2:
        return np.nan, np.nan, np.nan
    prediction = medians[bin_index]
    valid = np.isfinite(prediction)
    variance = np.nanvar(flux[valid])
    coherence = 1.0 - np.nanvar(flux[valid] - prediction[valid]) / variance if variance > 0 else 0.0
    semi_amplitude = 0.5 * (
        np.nanpercentile(medians[available], 95) - np.nanpercentile(medians[available], 5)
    )
    residual_sigma = _robust_sigma(flux[valid] - prediction[valid])
    amplitude_snr = semi_amplitude / residual_sigma if residual_sigma > 0 else np.inf
    return float(coherence), float(semi_amplitude), float(amplitude_snr)


def _sinusoid_r2(time, flux, period):
    """Fraction of variance explained by a sinusoid at the trial period."""
    angle = 2.0 * np.pi * time / period
    design = np.column_stack((np.ones(time.size), np.sin(angle), np.cos(angle)))
    coefficients, *_ = np.linalg.lstsq(design, flux, rcond=None)
    residual = flux - design @ coefficients
    variance = np.nanvar(flux)
    return float(1.0 - np.nanvar(residual) / variance) if variance > 0 else 0.0


def _block_median(time, flux, max_points=6000):
    """Reduce BLS cost while retaining eclipse-scale structure."""
    if time.size <= max_points:
        return time, flux
    block_size = int(np.ceil(time.size / max_points))
    usable = time.size - time.size % block_size
    binned_time = np.nanmedian(time[:usable].reshape(-1, block_size), axis=1)
    binned_flux = np.nanmedian(flux[:usable].reshape(-1, block_size), axis=1)
    if usable < time.size:
        binned_time = np.append(binned_time, np.nanmedian(time[usable:]))
        binned_flux = np.append(binned_flux, np.nanmedian(flux[usable:]))
    return binned_time, binned_flux


def analyze_lightcurve(
    path,
    *,
    min_rotation_period=0.1,
    max_rotation_period=13.0,
    min_eclipse_period=0.3,
    max_eclipse_period=13.0,
    eclipse_snr_threshold=12.0,
    eclipse_min_depth=0.001,
    eclipse_max_sinusoid_r2=0.5,
    rotation_fap_threshold=1e-5,
    rotation_coherence_threshold=0.2,
    rotation_amplitude_snr_threshold=1.5,
):
    """Measure BLS eclipse metrics and LS rotational-modulation metrics."""
    time, flux, flux_err = _load_lightcurve_arrays(path)
    baseline = float(np.ptp(time))
    if baseline <= 2 * min_rotation_period:
        raise ValueError(f"Light-curve baseline is too short: {baseline:.3g} day")

    rotation_max = min(float(max_rotation_period), baseline / 2)
    usable_error = np.isfinite(flux_err) & (flux_err > 0)
    if np.any(usable_error):
        typical_error = np.nanmedian(flux_err[usable_error])
        ls_error = np.where(usable_error, flux_err, typical_error)
    else:
        ls_error = None
    ls = LombScargle(time, flux, dy=ls_error)
    frequency, power = ls.autopower(
        minimum_frequency=1.0 / rotation_max,
        maximum_frequency=1.0 / min_rotation_period,
        samples_per_peak=10,
    )
    best_index = int(np.nanargmax(power))
    p_rot = float(1.0 / frequency[best_index])
    ls_power = float(power[best_index])
    try:
        ls_fap = float(ls.false_alarm_probability(ls_power, method="baluev"))
    except Exception:
        ls_fap = np.nan
    coherence, semi_amplitude, amplitude_snr = _phase_metrics(time, flux, p_rot)
    sinusoid_r2 = _sinusoid_r2(time, flux, p_rot)
    n_rotation_cycles = baseline / p_rot
    p_rot_err = p_rot**2 / baseline

    bls_time, bls_flux = _block_median(time, flux)
    point_noise = _robust_sigma(np.diff(bls_flux)) / np.sqrt(2.0)
    if not np.isfinite(point_noise) or point_noise <= 0:
        point_noise = _robust_sigma(bls_flux)
    eclipse_max = min(float(max_eclipse_period), baseline / 3)
    bls_values = {
        "bls_period_day": np.nan,
        "bls_duration_day": np.nan,
        "bls_depth": np.nan,
        "bls_snr": np.nan,
        "bls_duty_cycle": np.nan,
        "dip_asymmetry": np.nan,
        "eclipse_candidate": False,
    }
    if eclipse_max > min_eclipse_period:
        durations = np.array([0.03, 0.06, 0.12])
        durations = durations[durations < min_eclipse_period]
        bls = BoxLeastSquares(
            bls_time, bls_flux, dy=np.full_like(bls_flux, point_noise)
        )
        result = bls.autopower(
            durations,
            objective="snr",
            method="fast",
            minimum_n_transit=3,
            minimum_period=min_eclipse_period,
            maximum_period=eclipse_max,
            frequency_factor=3.0,
        )
        index = int(np.nanargmax(result.power))
        bls_period = _finite_float(result.period[index])
        bls_duration = _finite_float(result.duration[index])
        bls_depth = _finite_float(result.depth[index])
        bls_snr = _finite_float(result.power[index])
        duty_cycle = bls_duration / bls_period
        low_tail = np.nanmedian(flux) - np.nanpercentile(flux, 1)
        high_tail = np.nanpercentile(flux, 99) - np.nanmedian(flux)
        dip_asymmetry = low_tail / high_tail if high_tail > 0 else np.inf
        eclipse_candidate = bool(
            bls_snr >= eclipse_snr_threshold
            and bls_depth >= eclipse_min_depth
            and duty_cycle <= 0.2
            and dip_asymmetry >= 1.1
            and sinusoid_r2 <= eclipse_max_sinusoid_r2
        )
        bls_values = {
            "bls_period_day": bls_period,
            "bls_duration_day": bls_duration,
            "bls_depth": bls_depth,
            "bls_snr": bls_snr,
            "bls_duty_cycle": duty_cycle,
            "dip_asymmetry": dip_asymmetry,
            "eclipse_candidate": eclipse_candidate,
        }

    clear_rotation = bool(
        np.isfinite(ls_fap)
        and ls_fap <= rotation_fap_threshold
        and coherence >= rotation_coherence_threshold
        and amplitude_snr >= rotation_amplitude_snr_threshold
        and n_rotation_cycles >= 2
        and not bls_values["eclipse_candidate"]
    )
    return {
        "n_cadences_analyzed": int(time.size),
        "baseline_day": baseline,
        "p_rot_day": p_rot,
        "p_rot_resolution_day": p_rot_err,
        "ls_power": ls_power,
        "ls_fap": ls_fap,
        "phase_coherence": coherence,
        "rotation_semi_amplitude": semi_amplitude,
        "rotation_amplitude_snr": amplitude_snr,
        "sinusoid_r2": sinusoid_r2,
        "n_rotation_cycles": n_rotation_cycles,
        "clear_rotation": clear_rotation,
        **bls_values,
    }


def analyze_saved_lightcurves(manifest, output_root, **kwargs):
    """Analyze one saved light curve per Frame without displaying per-star plots."""
    selected = select_lightcurve_files(manifest, output_root)
    rows = []
    for _, item in tqdm(
        selected.iterrows(), total=len(selected), desc="TESS variability", unit="frame"
    ):
        base = {
            "frame_id": str(item["frame_id"]),
            "object_name": item.get("object_name"),
            "sector": item.get("sector"),
            "author": item.get("author"),
            "fits_path": str(item["resolved_fits_path"]),
        }
        try:
            rows.append({**base, "analysis_status": "ok", **analyze_lightcurve(
                item["resolved_fits_path"], **kwargs
            )})
        except Exception as exc:
            rows.append({**base, "analysis_status": "error", "analysis_error": repr(exc)})
    return pd.DataFrame.from_records(rows)


def load_stellar_rotation_parameters(
    frame_ids,
    *,
    analysis_root=None,
    spec_label="orders03-08",
    iso_label="mistfit_g_st38",
):
    """Load R and v sin(i) using the canonical ``rot_evol`` result loaders."""
    from rot_evol import (
        ANALYSIS_ROOT,
        find_frame_dir,
        normalize_frame_id,
        posterior_mean_std,
        result_path,
    )

    if analysis_root is None:
        analysis_root = ANALYSIS_ROOT

    columns = [
        "frame_id", "frame_norm", "stellar_parameter_status",
        "radius_rsun", "radius_err_rsun", "vsini_km_s", "vsini_err_km_s",
        "p_2piR_vsini_day", "p_2piR_vsini_err_day", "stellar_parameter_error",
    ]
    rows = []
    for original_frame_id in tqdm(frame_ids, desc="R and v sin(i)", unit="frame"):
        frame_id = normalize_frame_id(original_frame_id)
        base = {"frame_id": str(original_frame_id), "frame_norm": frame_id}
        try:
            frame_dir = find_frame_dir(Path(analysis_root), frame_id)
            spec_nc = result_path(frame_dir, spec_label)
            iso_nc = result_path(frame_dir, iso_label)
            vsini, vsini_err = posterior_mean_std(spec_nc, "vsini")
            radius, radius_err = posterior_mean_std(iso_nc, "radius")
            p_vsini = 2 * np.pi * R_SUN_KM * radius / vsini / DAY_S
            p_vsini_err = p_vsini * np.hypot(radius_err / radius, vsini_err / vsini)
            rows.append({
                **base,
                "stellar_parameter_status": "ok",
                "radius_rsun": radius,
                "radius_err_rsun": radius_err,
                "vsini_km_s": vsini,
                "vsini_err_km_s": vsini_err,
                "p_2piR_vsini_day": p_vsini,
                "p_2piR_vsini_err_day": p_vsini_err,
            })
        except Exception as exc:
            rows.append({
                **base,
                "stellar_parameter_status": "error",
                "stellar_parameter_error": repr(exc),
            })
    return pd.DataFrame.from_records(rows, columns=columns)


def save_rotation_comparison(variability, stellar, path):
    """Save P_rot versus 2*pi*R/vsini for clear non-eclipsing rotators."""
    import matplotlib.pyplot as plt

    table = variability.merge(stellar, on="frame_id", how="left")
    usable = table.loc[
        table["analysis_status"].eq("ok")
        & table["clear_rotation"].fillna(False)
        & ~table["eclipse_candidate"].fillna(False)
        & table["stellar_parameter_status"].eq("ok")
    ].copy()
    usable["p_rot_over_p_2piR_vsini"] = (
        usable["p_rot_day"] / usable["p_2piR_vsini_day"]
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    if usable.empty:
        axis.text(0.5, 0.5, "No clear rotators with R and v sin(i)", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.errorbar(
            usable["p_rot_day"], usable["p_2piR_vsini_day"],
            xerr=usable["p_rot_resolution_day"],
            yerr=usable["p_2piR_vsini_err_day"],
            fmt="o", ms=5, alpha=0.8, capsize=2,
        )
        limits = np.array([
            usable["p_rot_day"].min(), usable["p_rot_day"].max(),
            usable["p_2piR_vsini_day"].min(), usable["p_2piR_vsini_day"].max(),
        ])
        lower = max(0.05, float(np.nanmin(limits)) * 0.75)
        upper = float(np.nanmax(limits)) * 1.35
        axis.plot([lower, upper], [lower, upper], "--", color="0.35", label="equality")
        for _, row in usable.iterrows():
            label = row.get("object_name")
            if pd.isna(label) or not str(label).strip():
                label = row["frame_id"]
            axis.annotate(str(label), (row["p_rot_day"], row["p_2piR_vsini_day"]),
                          xytext=(4, 3), textcoords="offset points", fontsize=7)
        axis.set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper))
        axis.set_xlabel(r"Photometric $P_{\rm rot}$ [day]")
        axis.set_ylabel(r"$2\pi R/(v\sin i)$ [day]")
        axis.set_title("Clear rotational modulation")
        axis.legend(loc="best")
        axis.grid(alpha=0.25, which="both")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return usable, path
