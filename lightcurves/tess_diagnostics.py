"""TPF download and three-panel TESS diagnostic figures."""

from pathlib import Path
import shutil

import astropy.units as u
import numpy as np


def load_or_download_tpf(
    coordinate, sector, frame_dir, cache_dir, *, search_radius=5 * u.arcsec, cutout_size=15
):
    """Return a pipeline TPF, falling back to a TESScut target pixel cutout."""
    import lightkurve as lk

    sector_number = int(sector) if sector is not None and np.isfinite(sector) else None
    token = f"sector_{sector_number:04d}" if sector_number else "sector_unknown"
    tpf_path = Path(frame_dir) / f"{token}_tpf.fits"
    if tpf_path.is_file():
        return lk.read(tpf_path), "cached", tpf_path, None
    errors = []
    kwargs = {"mission": "TESS", "radius": search_radius}
    if sector_number is not None:
        kwargs["sector"] = sector_number
    try:
        search = lk.search_targetpixelfile(coordinate, **kwargs)
        if len(search):
            tpf = search[0].download(download_dir=str(cache_dir))
            if tpf is not None:
                _save_tpf(tpf, tpf_path)
                return tpf, "pipeline_tpf", tpf_path, None
    except Exception as exc:
        errors.append(f"pipeline TPF: {exc!r}")
    try:
        kwargs = {"sector": sector_number} if sector_number is not None else {}
        search = lk.search_tesscut(coordinate, **kwargs)
        if len(search):
            tpf = search[0].download(
                cutout_size=(cutout_size, cutout_size), download_dir=str(cache_dir)
            )
            if tpf is not None:
                _save_tpf(tpf, tpf_path)
                return tpf, "tesscut", tpf_path, None
    except Exception as exc:
        errors.append(f"TESScut: {exc!r}")
    return None, "not_available", None, "; ".join(errors) or "No TPF product found"


def _save_tpf(tpf, destination):
    source_value = getattr(tpf, "path", None) or getattr(tpf, "filename", None)
    source = Path(source_value) if source_value else None
    if source and source.is_file():
        shutil.copy2(source, destination)
    else:
        tpf.to_fits(path=destination, overwrite=True)


def save_diagnostic_figure(cleaned, tpf, path, title, *, dpi=160):
    """Save TPF, light curve, Lomb--Scargle, and LS-folded curve in one figure."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_tpf, ax_lc, ax_ls, ax_phase = axes.ravel()
    if tpf is None:
        ax_tpf.text(0.5, 0.5, "TPF not available", ha="center", va="center")
        ax_tpf.set_axis_off()
    else:
        cube = np.asarray(getattr(tpf.flux, "value", tpf.flux), dtype=float)
        image = np.nanmedian(cube, axis=0)
        finite = image[np.isfinite(image)]
        vmin, vmax = np.nanpercentile(finite, [5, 99.5]) if finite.size else (None, None)
        shown = ax_tpf.imshow(
            image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        figure.colorbar(shown, ax=ax_tpf, label="Median flux")
        mask = getattr(tpf, "pipeline_mask", None)
        if mask is not None and np.shape(mask) == np.shape(image) and np.any(mask):
            ax_tpf.contour(np.asarray(mask, dtype=float), levels=[0.5], colors="white")
        ax_tpf.set(xlabel="Pixel column", ylabel="Pixel row")
    ax_tpf.set_title("TESS target pixel file")

    time = np.asarray(cleaned.time.value, dtype=float)
    flux = np.asarray(getattr(cleaned.flux, "value", cleaned.flux), dtype=float)
    ax_lc.plot(time, flux, ".", ms=1.5, alpha=0.65, rasterized=True)
    ax_lc.set(
        title="Normalized light curve",
        xlabel=f"Time [{getattr(cleaned.time, 'format', 'day')}]",
        ylabel="Normalized flux",
    )

    periodogram = cleaned.to_periodogram(method="lombscargle")
    period = np.asarray(periodogram.period.to_value(u.day), dtype=float)
    power = np.asarray(getattr(periodogram.power, "value", periodogram.power), dtype=float)
    valid = np.isfinite(period) & np.isfinite(power) & (period > 0)
    if not np.any(valid):
        raise ValueError("Lomb--Scargle periodogram has no finite positive periods")
    best_index = np.flatnonzero(valid)[np.nanargmax(power[valid])]
    best_period = float(period[best_index])
    order = np.argsort(period[valid])
    ax_ls.plot(period[valid][order], power[valid][order], color="tab:blue", lw=0.8)
    ax_ls.axvline(
        best_period, color="tab:red", ls="--", lw=1, label=f"Peak = {best_period:.4g} d"
    )
    ax_ls.set_xscale("log")
    ax_ls.set(
        title="Lomb--Scargle periodogram", xlabel="Period [day]", ylabel="Power"
    )
    ax_ls.legend(loc="best")

    phase = np.mod(time - time[0], best_period) / best_period
    ax_phase.plot(phase, flux, ".", ms=1.5, alpha=0.35, rasterized=True)
    edges = np.linspace(0.0, 1.0, 31)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.full(centers.size, np.nan)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (phase >= left) & (phase < right)
        if np.any(in_bin):
            binned[index] = np.nanmedian(flux[in_bin])
    finite_bins = np.isfinite(binned)
    ax_phase.plot(
        centers[finite_bins], binned[finite_bins], "o-", color="tab:red", ms=3, lw=1,
        label="Phase-bin median",
    )
    ax_phase.set(
        title=f"Phase folded at LS peak ({best_period:.5g} d)",
        xlabel="Phase", ylabel="Normalized flux", xlim=(0, 1),
    )
    ax_phase.legend(loc="best")
    figure.suptitle(title)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return best_period
