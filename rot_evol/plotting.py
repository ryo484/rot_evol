"""Result serialization and diagnostic plotting."""

from pathlib import Path

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

from .data import RESULT_ROOT
from .models import (
    DAY_S,
    R_SUN_KM,
    evaluate_rotation_law,
    jaxspin_rotation_law,
    normalize_rotation_law,
    pi0_linear,
    rotation_slope_names,
)

def save_isochrone_mean_triangle(means, output_path, *, dpi=200):
    """Save a scatter-plot triangle of per-star isochrone posterior means."""
    required = {"frame_id", "mass", "age", "feh", "radius"}
    missing = required.difference(means.columns)
    if missing:
        raise ValueError(f"Missing isochrone-mean columns: {sorted(missing)}")
    if len(means) < 2:
        raise ValueError("At least two stars are required for a triangle plot")

    columns = ("mass", "age", "feh", "radius")
    values = means.loc[:, columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Isochrone posterior means must be finite")

    plot_range = []
    for column in values.T:
        lower = float(np.min(column))
        upper = float(np.max(column))
        if lower == upper:
            padding = max(abs(lower) * 0.05, 0.05)
            lower -= padding
            upper += padding
        plot_range.append((lower, upper))

    figure = corner.corner(
        values,
        bins=min(20, max(5, int(np.sqrt(len(means)) * 2))),
        range=plot_range,
        labels=(r"$M/M_\odot$", "Age [Gyr]", "[Fe/H]", r"$R/R_\odot$"),
        color="C0",
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        data_kwargs={"ms": 4, "alpha": 0.75},
        hist_kwargs={"histtype": "step", "linewidth": 1.5},
        quiet=True,
    )
    figure.suptitle(
        f"Isochrone posterior means (N={len(means)})",
        y=1.02,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path

def selection_label(mass_range=None, feh_range=None, radius_range=None):
    """Build a result label from mass, metallicity, and radius selections."""
    parts = []
    for name, bounds in (
        ("mass", mass_range),
        ("feh", feh_range),
        ("radius", radius_range),
    ):
        if bounds is None:
            continue
        if len(bounds) != 2 or all(bound is None for bound in bounds):
            raise ValueError(f"{name}_range must contain at least one finite bound")
        lower, upper = bounds
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{name}_range must be an ordered (lower, upper) pair")
        if lower is None:
            parts.append(f"{name}{upper:g}low")
        elif upper is None:
            parts.append(f"{name}{lower:g}up")
        else:
            parts.append(f"{name}{lower:g}-{upper:g}")
    return "_".join(parts) or "all"

def save_logage_histogram(
    re,
    output_path,
    idata=None,
    bins="auto",
    dpi=150,
):
    """Save the observed age distribution and its fixed or inferred Normal model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logage = np.asarray(re.logage, dtype=float)
    median = re.logage_prior_loc
    std = re.logage_prior_scale

    figure, axis = plt.subplots(figsize=(6, 4))
    _, bin_edges, _ = axis.hist(
        logage,
        bins=bins,
        density=True,
        color="C0",
        alpha=0.65,
        label="Per-star medians",
    )
    inferred_names = None
    names = ("lognorm_age_mu", "lognorm_age_sigma")
    if idata is not None and set(names).issubset(idata.posterior.data_vars):
        inferred_names = names
    if inferred_names is not None:
        mu = np.asarray(idata.posterior[inferred_names[0]], dtype=float).reshape(-1)
        sigma = np.asarray(idata.posterior[inferred_names[1]], dtype=float).reshape(-1)
        model_mu = float(np.median(mu))
        model_sigma = float(np.median(sigma))
        x_limits = np.array(
            [model_mu - 4.0 * model_sigma, model_mu + 4.0 * model_sigma]
        )
        x = np.linspace(*x_limits, 500)
        pdf = np.exp(-0.5 * ((x[None, :] - mu[:, None]) / sigma[:, None]) ** 2)
        pdf /= sigma[:, None] * np.sqrt(2 * np.pi)
        lower, model_median, upper = np.quantile(pdf, [0.16, 0.5, 0.84], axis=0)
        axis.fill_between(
            x, lower, upper, color="C1", alpha=0.20,
            label="Inferred Normal: 68% interval",
        )
        axis.plot(x, model_median, color="C1", lw=2, label="Inferred Normal: median")
        axis.axvline(
            model_mu, color="C1", lw=1.5, ls="--",
            label=fr"posterior median $\mu$ = {model_mu:.4f}",
        )
    else:
        x_limits = np.array([median - 4.0 * std, median + 4.0 * std])
        x = np.linspace(*x_limits, 500)
        normal_pdf = np.exp(-0.5 * ((x - median) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        axis.plot(x, normal_pdf, color="C1", lw=2, label="Shared Normal prior")
        axis.axvline(median, color="C1", lw=1.5, ls="--", label=f"median = {median:.4f}")
    axis.set(
        xlabel=r"Per-star median $\log_{10}(\mathrm{age/yr})$",
        ylabel="Density",
        title=f"Age distribution of {re.N} stars",
        xlim=x_limits,
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_logage_posterior_histogram(
    re,
    idata,
    output_path,
    bins=50,
    dpi=150,
):
    """Plot posterior stellar logages and the inferred population density."""
    required = {"logage", "lognorm_age_mu", "lognorm_age_sigma"}
    missing = required.difference(idata.posterior.data_vars)
    if missing:
        raise KeyError(
            "Missing posterior variables for the logage population plot: "
            f"{sorted(missing)}"
        )

    logage_draws = np.asarray(idata.posterior["logage"], dtype=float)
    if logage_draws.ndim < 3:
        raise ValueError(
            "Posterior logage must have chain, draw, and star dimensions"
        )
    # One representative posterior age per star; do not pool all MCMC draws.
    logage = np.median(logage_draws, axis=(0, 1)).reshape(-1)
    mu = np.asarray(idata.posterior["lognorm_age_mu"], dtype=float).reshape(-1)
    sigma = np.asarray(
        idata.posterior["lognorm_age_sigma"], dtype=float
    ).reshape(-1)
    logage = logage[np.isfinite(logage)]
    valid_hyper = np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    mu = mu[valid_hyper]
    sigma = sigma[valid_hyper]
    if logage.size == 0 or mu.size == 0:
        raise ValueError("No finite posterior draws are available for the logage plot")

    limits = np.array(
        [
            min(np.quantile(logage, 0.001), np.min(mu - 4.0 * sigma)),
            max(np.quantile(logage, 0.999), np.max(mu + 4.0 * sigma)),
        ]
    )
    x = np.linspace(*limits, 500)
    population_pdf = np.exp(
        -0.5 * ((x[None, :] - mu[:, None]) / sigma[:, None]) ** 2
    )
    population_pdf /= sigma[:, None] * np.sqrt(2.0 * np.pi)
    lower, median, upper = np.quantile(
        population_pdf, [0.16, 0.5, 0.84], axis=0
    )

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(
        logage,
        bins=bins,
        range=limits,
        density=True,
        color="0.35",
        alpha=0.35,
        label="per-star posterior median",
    )
    axis.fill_between(
        x,
        lower,
        upper,
        color="C1",
        alpha=0.25,
        label="inferred population: 68% credible interval",
    )
    axis.plot(
        x,
        median,
        color="C1",
        lw=2,
        label="inferred population: median density",
    )
    axis.set(
        xlim=limits,
        xlabel=r"Posterior $\log_{10}(\mathrm{age/yr})$",
        ylabel="Probability density",
        title=f"Posterior stellar ages and population distribution (N={re.N})",
    )
    axis.legend(fontsize="small")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path

def _output_dir(label, result_root=RESULT_ROOT):
    """Create the output directory, rejecting labels outside result_root."""
    label = Path(label)
    if label.is_absolute() or ".." in label.parts or str(label) in ("", "."):
        raise ValueError("label must be a non-empty relative path without '..'")
    output_dir = Path(result_root) / label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def save_inference_data(
    re,
    idata,
    label,
    *,
    result_root=RESULT_ROOT,
):
    """Add metadata and save an InferenceData object as ``mcmc.nc``."""
    idata.attrs["frame_ids"] = ",".join(re.frame_id)
    posterior_names = set(idata.posterior.data_vars)
    idata.attrs.setdefault(
        "rotation_law",
        "broken_power" if {"a1", "a2"}.issubset(posterior_names) else "power",
    )
    idata.attrs["model"] = idata.attrs["rotation_law"]
    path = _output_dir(label, result_root) / "mcmc.nc"
    idata.to_netcdf(path)
    return path

def _sample_values_first(samples, name):
    """Return a posterior variable with the stacked sample axis first."""
    values = samples[name]
    other_dims = [dimension for dimension in values.dims if dimension != "sample"]
    return np.asarray(values.transpose("sample", *other_dims), dtype=float)


def _corner_samples(samples, var_names):
    """Build a 2-D corner array, expanding vector-valued parameters."""
    columns = []
    labels = []
    plotted_var_names = []
    for name in var_names:
        values = _sample_values_first(samples, name)
        values = values.reshape(values.shape[0], -1)
        kept_for_variable = False
        for index, column in enumerate(values.T):
            finite = column[np.isfinite(column)]
            if finite.size == 0 or np.min(finite) == np.max(finite):
                continue
            columns.append(column)
            labels.append(name if values.shape[1] == 1 else f"{name}[{index}]")
            kept_for_variable = True
        if kept_for_variable:
            plotted_var_names.append(name)
    if not columns:
        raise ValueError("No varying finite posterior parameters are available to plot")
    return np.column_stack(columns), labels, plotted_var_names

def infer_rotation_law(posterior):
    """Infer the rotation law from posterior variable names."""
    names = set(posterior.data_vars)
    if {"a1", "a2"}.issubset(names):
        return "broken_power"
    if {"gp_latent", "gp_amplitude", "gp_scale"}.issubset(names):
        return "gp"
    if {"Ro_wmb_factor", "feh"}.issubset(names):
        return "jaxspin"
    if "a" in names:
        return "power"
    raise KeyError("Posterior does not contain a recognized rotation law")

def posterior_rotation_draws(
    age, samples, rotation_law=None, break_age=None, *, mass=None, max_draws=200
):
    """Evaluate a rotation relation for stacked posterior samples."""
    rotation_law = normalize_rotation_law(
        rotation_law or infer_rotation_law(samples)
    )
    if rotation_law == "jaxspin":
        if mass is None:
            raise ValueError("mass is required for jaxspin posterior draws")
        ro_wmb_factor = _sample_values_first(samples, "Ro_wmb_factor")
        feh = _sample_values_first(samples, "feh")
        count = min(ro_wmb_factor.size, max_draws)
        indices = np.linspace(0, ro_wmb_factor.size - 1, count, dtype=int)
        evaluation_age = np.asarray(age, dtype=float)
        if evaluation_age.ndim > 1 and evaluation_age.shape[0] == 1:
            evaluation_age = evaluation_age[0]
        return np.stack([
            np.asarray(jaxspin_rotation_law(
                evaluation_age,
                float(mass),
                float(feh[index]),
                float(ro_wmb_factor[index]),
                n_age=2001,
            ))
            for index in indices
        ])
    if rotation_law == "gp":
        slopes = (
            _sample_values_first(samples, "gp_latent")[:, None, :],
            _sample_values_first(samples, "gp_amplitude")[:, None],
            _sample_values_first(samples, "gp_scale")[:, None],
        )
        b = _sample_values_first(samples, "b")[:, None]
    else:
        slopes = tuple(
            np.asarray(samples[name].values)[:, None]
            for name in rotation_slope_names(rotation_law)
        )
        b = np.asarray(samples["b"].values)[:, None]
    if rotation_law == "broken_power" and break_age is None:
        if "break_age" not in samples:
            raise KeyError("broken_power posterior is missing break_age")
        break_age = np.asarray(samples["break_age"].values)[:, None]
    return np.asarray(
        evaluate_rotation_law(age, rotation_law, slopes, b, break_age)
    )

def _conditional_age_vsini_samples(
    re,
    idata,
    isochrone_posterior,
    *,
    predict_P_rot=False,
    rotation_law=None,
    break_age=None,
    max_posterior_draws=1000,
    max_isochrone_samples=2000,
    seed=0,
):
    """Draw per-star age and vsini conditional on the marginalized posterior."""
    if not np.array_equal(re.frame_id, isochrone_posterior.frame_id):
        raise ValueError("RotEvol and IsochronePosterior frame_id/order must match")
    rotation_law = normalize_rotation_law(
        rotation_law or infer_rotation_law(idata.posterior)
    )
    predict_age_dist = {
        "lognorm_age_mu", "lognorm_age_sigma"
    }.issubset(idata.posterior.data_vars)
    required = {*rotation_slope_names(rotation_law), "cosi"}
    if rotation_law != "jaxspin":
        required.add("b")
    if predict_age_dist:
        required.update({"lognorm_age_mu", "lognorm_age_sigma"})
    if rotation_law == "broken_power" and break_age is None:
        required.add("break_age")
    missing = required.difference(idata.posterior.data_vars)
    if missing:
        raise KeyError(f"Missing posterior variables for conditional plots: {sorted(missing)}")

    samples = idata.posterior.stack(sample=("chain", "draw"))
    rng = np.random.default_rng(seed)
    ndraws = samples.sizes["sample"]
    if rotation_law == "jaxspin":
        max_posterior_draws = min(max_posterior_draws, 200)
    draw_indices = np.sort(
        rng.choice(ndraws, size=min(ndraws, max_posterior_draws), replace=False)
    )
    niso = isochrone_posterior.Nsamples
    iso_indices = np.sort(
        rng.choice(niso, size=min(niso, max_isochrone_samples), replace=False)
    )

    slopes = tuple(
        _sample_values_first(samples, name)[draw_indices]
        for name in rotation_slope_names(rotation_law)
    )
    b = (
        None
        if rotation_law == "jaxspin"
        else _sample_values_first(samples, "b")[draw_indices]
    )
    if rotation_law == "broken_power" and break_age is None:
        break_age_draws = _sample_values_first(
            samples, "break_age"
        )[draw_indices]
    else:
        break_age_draws = break_age
    if predict_age_dist:
        mu = _sample_values_first(samples, "lognorm_age_mu")[draw_indices]
        sigma = _sample_values_first(samples, "lognorm_age_sigma")[draw_indices]
    cosi = _sample_values_first(samples, "cosi")[draw_indices]
    if cosi.shape != (draw_indices.size, re.N):
        raise ValueError(
            f"Expected cosi shape {(draw_indices.size, re.N)}, got {cosi.shape}"
        )

    age_draws = np.empty((re.N, draw_indices.size))
    vsini_draws = np.empty_like(age_draws)
    for star in range(re.N):
        age = np.asarray(isochrone_posterior.age[star, iso_indices], dtype=float)
        if predict_age_dist:
            logage = np.log10(age) + 9.0
            log_population_age = (
                -0.5 * ((logage[None, :] - mu[:, None]) / sigma[:, None]) ** 2
                - np.log(sigma[:, None])
                - np.log(age[None, :])
            )
            logweight = log_population_age - np.log(
                np.asarray(pi0_linear(age), dtype=float)[None, :]
            )
        else:
            logweight = np.zeros((draw_indices.size, age.size))
        if rotation_law == "jaxspin":
            rotation_relation = np.stack([
                np.asarray(jaxspin_rotation_law(
                    age,
                    float(np.median(re.mass)),
                    float(slopes[1][draw]),
                    float(slopes[0][draw]),
                    n_age=2001,
                ))
                for draw in range(draw_indices.size)
            ])
        else:
            rotation_relation = np.asarray(
                evaluate_rotation_law(
                    age[None, :],
                    rotation_law,
                    tuple(slope[:, None] for slope in slopes),
                    b[:, None],
                    (
                        break_age_draws[:, None]
                        if rotation_law == "broken_power" and break_age is None
                        else break_age_draws
                    ),
                )
            )
        if predict_P_rot:
            radius = np.asarray(
                isochrone_posterior.radius[star, iso_indices], dtype=float
            )
            v = (
                2.0
                * np.pi
                * radius[None, :]
                * R_SUN_KM
                / (rotation_relation * DAY_S)
            )
        else:
            v = rotation_relation
        vsini = v * np.sqrt(1.0 - cosi[:, star, None] ** 2)
        loglike = -0.5 * (
            (re.vsini[star] - vsini) / re.vsini_err[star]
        ) ** 2
        log_probability = loglike + logweight
        log_probability -= np.max(log_probability, axis=1, keepdims=True)
        probability = np.exp(log_probability)
        probability /= np.sum(probability, axis=1, keepdims=True)
        cdf = np.cumsum(probability, axis=1)
        choices = np.sum(cdf < rng.random((draw_indices.size, 1)), axis=1)
        choices = np.minimum(choices, age.size - 1)
        rows = np.arange(draw_indices.size)
        age_draws[star] = age[choices]
        vsini_draws[star] = vsini[rows, choices]
    return age_draws, vsini_draws

def save_figures(
    re,
    idata,
    label,
    *,
    isochrone_posterior=None,
    predict_P_rot=False,
    rotation_law=None,
    break_age=None,
    result_root=RESULT_ROOT,
    dpi=200,
):
    """Save corner, trace, and rotation-relation figures."""
    output_dir = _output_dir(label, result_root)
    paths = {
        "corner": output_dir / "corner.png",
        "trace": output_dir / "trace.png",
        "rotation": output_dir / "rotation.png",
        "rotation_linear": output_dir / "rotation_linear.png",
        "vsini_obs_vs_post": output_dir / "vsini_obs_vs_post.png",
        "age_obs_vs_post": output_dir / "age_obs_vs_post.png",
    }
    if not predict_P_rot:
        paths["rotation_inverse_linear"] = output_dir / "rotation_inverse_linear.png"
    samples = idata.posterior.stack(sample=("chain", "draw"))
    rotation_law = normalize_rotation_law(
        rotation_law or infer_rotation_law(idata.posterior)
    )
    # Plot only globally sampled model parameters. ``b`` is deterministic
    # (b = 10**logb), so the sampled parameter shown here is ``logb``.
    var_names = [
        name for name in rotation_slope_names(rotation_law)
        if name != "gp_latent"
    ]
    if rotation_law != "jaxspin":
        var_names.append("logb")
    if rotation_law == "broken_power" and "break_age" in idata.posterior:
        var_names.append("break_age")
    age_dist_vars = ["lognorm_age_mu", "lognorm_age_sigma"]
    var_names.extend(
        name for name in age_dist_vars if name in idata.posterior.data_vars
    )
    corner_values, corner_labels, _ = _corner_samples(samples, var_names)

    if corner_values.shape[1] == 1:
        values = corner_values[:, 0]
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.hist(values, bins=min(30, max(1, np.unique(values).size)), color="C0", alpha=0.75)
        axis.axvline(np.median(values), color="black", linestyle="--", label="median")
        axis.set(xlabel=corner_labels[0], ylabel="Posterior samples")
        axis.legend()
        figure.tight_layout()
    else:
        figure = corner.corner(
            corner_values,
            labels=corner_labels,
            show_titles=True,
            title_fmt=".3f",
            quantiles=[0.16, 0.5, 0.84],
        )
    plt.tight_layout()
    figure.savefig(paths["corner"], dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    axes = az.plot_trace(idata, var_names=var_names, compact=False)
    figure = np.asarray(axes).flat[0].figure
    figure.tight_layout()
    figure.savefig(paths["trace"], dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    age_grid = np.geomspace(
        max(0.01, np.min(re.age - re.age_err) * 0.8),
        min(13.8, np.max(re.age + re.age_err) * 1.2),
        300,
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    if predict_P_rot:
        if isochrone_posterior is None:
            raise ValueError(
                "isochrone_posterior is required to plot P/sin(i)"
            )
        if not np.array_equal(re.frame_id, isochrone_posterior.frame_id):
            raise ValueError("RotEvol and IsochronePosterior frame_id/order must match")
        radius = np.median(isochrone_posterior.radius, axis=1)
        radius_err = np.std(isochrone_posterior.radius, axis=1)
        period_over_sini = (
            2.0 * np.pi * radius * R_SUN_KM / (re.vsini * DAY_S)
        )
        period_over_sini_err = period_over_sini * np.sqrt(
            (radius_err / radius) ** 2 + (re.vsini_err / re.vsini) ** 2
        )
        axis.errorbar(
            re.age,
            period_over_sini,
            xerr=re.age_err,
            yerr=period_over_sini_err,
            fmt="o",
            color="black",
            ecolor="0.65",
            capsize=2,
            label=r"observed $P/\sin i$",
        )
    else:
        axis.errorbar(
            re.age,
            re.vsini,
            xerr=re.age_err,
            yerr=re.vsini_err,
            fmt="o",
            color="black",
            ecolor="0.65",
            capsize=2,
            label=r"observed age and $v\sin i$",
        )
    model_draws = posterior_rotation_draws(
        age_grid[None, :],
        samples,
        rotation_law,
        break_age,
        mass=float(np.median(re.mass)),
    )
    lower, median, upper = np.percentile(model_draws, [16, 50, 84], axis=0)
    axis.fill_between(age_grid, lower, upper, color="C0", alpha=0.25, label="68% credible interval")
    axis.plot(age_grid, median, color="C0", lw=2, label=f"median {rotation_law}")
    ylabel = "Rotation period [day]" if predict_P_rot else r"Velocity [km s$^{-1}$]"
    axis.set(xscale="log", yscale="log", xlabel="Age [Gyr]", ylabel=ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["rotation"], dpi=dpi, bbox_inches="tight")
    axis.set(xscale="linear", yscale="linear", xlim = (0, 13), ylim = (0, 70))
    figure.tight_layout()
    figure.savefig(paths["rotation_linear"], dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    if not predict_P_rot:
        valid_observation = (
            np.isfinite(re.age) & np.isfinite(re.age_err)
            & np.isfinite(re.vsini) & np.isfinite(re.vsini_err)
            & (re.vsini > 0)
        )
        inverse_vsini = 1.0 / re.vsini[valid_observation]
        inverse_vsini_err = re.vsini_err[valid_observation] / re.vsini[valid_observation] ** 2
        inverse_model_draws = np.divide(
            1.0, model_draws, out=np.full_like(model_draws, np.nan, dtype=float),
            where=np.isfinite(model_draws) & (model_draws > 0),
        )
        inverse_lower, inverse_median, inverse_upper = np.nanpercentile(
            inverse_model_draws, [16, 50, 84], axis=0
        )

        inverse_figure, inverse_axis = plt.subplots(figsize=(7, 5))
        inverse_axis.errorbar(
            re.age[valid_observation], inverse_vsini,
            xerr=re.age_err[valid_observation], yerr=inverse_vsini_err,
            fmt="o", color="black", ecolor="0.65",
            capsize=2, label=r"observed $1/(v\sin i)$",
        )
        inverse_axis.fill_between(
            age_grid, inverse_lower, inverse_upper, color="C0", alpha=0.25,
            label="68% credible interval",
        )
        inverse_axis.plot(
            age_grid, inverse_median, color="C0", lw=2,
            label=f"median inverse {rotation_law}",
        )
        inverse_axis.set(
            xscale="linear", yscale="linear", xlim=(0, 13), ylim=(0, 1),
            xlabel="Age [Gyr]", ylabel=r"Inverse velocity [s km$^{-1}$]",
        )
        inverse_axis.legend()
        inverse_figure.tight_layout()
        inverse_figure.savefig(
            paths["rotation_inverse_linear"], dpi=dpi, bbox_inches="tight"
        )
        plt.close(inverse_figure)

    if {"age", "vsini"}.issubset(idata.posterior.data_vars):
        posterior_age = _sample_values_first(samples, "age").T
        posterior_vsini = _sample_values_first(samples, "vsini").T
    else:
        if isochrone_posterior is None:
            raise ValueError(
                "isochrone_posterior is required to plot the marginalized model"
            )
        posterior_age, posterior_vsini = _conditional_age_vsini_samples(
            re,
            idata,
            isochrone_posterior,
            predict_P_rot=predict_P_rot,
            rotation_law=rotation_law,
            break_age=break_age,
        )

    if {"age", "v"}.issubset(idata.posterior.data_vars):
        paths["age_post_vs_v_post"] = output_dir / "age_post_vs_v_post.png"
        posterior_v = _sample_values_first(samples, "v").T
        if posterior_v.shape != posterior_age.shape:
            raise ValueError(
                "Posterior age and v arrays must have the same (star, sample) shape"
            )
        age_lower, age_median, age_upper = np.percentile(
            posterior_age, [16, 50, 84], axis=1
        )
        v_lower, v_median, v_upper = np.percentile(
            posterior_v, [16, 50, 84], axis=1
        )

        posterior_figure, posterior_axis = plt.subplots(figsize=(7, 5))
        # Thin paired draws retain the within-star age--velocity covariance.
        draw_stride = max(1, posterior_age.shape[1] // 100)
        posterior_axis.scatter(
            posterior_age[:, ::draw_stride].ravel(),
            posterior_v[:, ::draw_stride].ravel(),
            s=5,
            color="0.45",
            alpha=0.08,
            linewidths=0,
            label="paired posterior draws",
        )
        posterior_axis.errorbar(
            age_median,
            v_median,
            xerr=[age_median - age_lower, age_upper - age_median],
            yerr=[v_median - v_lower, v_upper - v_median],
            fmt="o",
            color="black",
            ecolor="0.55",
            capsize=2,
            label="per-star posterior median and 68% interval",
        )
        posterior_axis.fill_between(
            age_grid,
            lower,
            upper,
            color="C0",
            alpha=0.25,
            label="rotation model: 68% credible interval",
        )
        posterior_axis.plot(
            age_grid,
            median,
            color="C0",
            lw=2,
            label=f"rotation model: median {rotation_law}",
        )
        posterior_axis.set(
            xscale="log",
            yscale="log",
            xlabel="Posterior age [Gyr]",
            ylabel=r"Posterior equatorial velocity [km s$^{-1}$]",
        )
        posterior_axis.legend(fontsize="small")
        posterior_figure.tight_layout()
        posterior_figure.savefig(
            paths["age_post_vs_v_post"], dpi=dpi, bbox_inches="tight"
        )
        plt.close(posterior_figure)

    comparisons = (
        (
            "vsini_obs_vs_post",
            re.vsini,
            re.vsini_err,
            posterior_vsini,
            r"Observed $v\sin i$ [km s$^{-1}$]",
            r"Posterior $v\sin i$ [km s$^{-1}$]",
        ),
        (
            "age_obs_vs_post",
            re.age,
            re.age_err,
            posterior_age,
            "Observed age [Gyr]",
            "Posterior age [Gyr]",
        ),
    )
    for name, observed, observed_err, posterior, xlabel, ylabel in comparisons:
        lower, median, upper = np.percentile(posterior, [16, 50, 84], axis=1)
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.errorbar(
            observed,
            median,
            xerr=observed_err,
            yerr=[median - lower, upper - median],
            fmt="o",
            color="C0",
            ecolor="0.6",
            capsize=2,
        )
        endpoints = np.concatenate(
            [observed - observed_err, observed + observed_err, lower, upper]
        )
        positive = endpoints[np.isfinite(endpoints) & (endpoints > 0)]
        if positive.size == 0:
            raise ValueError(f"{name} has no positive finite values for log axes")
        log_limits = np.log10([np.min(positive), np.max(positive)])
        log_span = max(log_limits[1] - log_limits[0], 0.1)
        limits = 10 ** np.array(
            [log_limits[0] - 0.05 * log_span, log_limits[1] + 0.05 * log_span]
        )
        axis.set(
            xlabel=xlabel, ylabel=ylabel, xlim=limits, ylim=limits,
            xscale="log", yscale="log",
        )
        axis.plot(
            limits, limits, color="black", linestyle="--", alpha=0.6,
            label="1:1", zorder=0,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths[name], dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return paths
