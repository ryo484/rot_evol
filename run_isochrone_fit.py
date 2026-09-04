#!/usr/bin/env python
"""Run and plot joint isochrone + rotation-law fits for selected populations."""

import argparse
from itertools import product
from pathlib import Path

import arviz as az
import numpyro

# Expose one host CPU device per chain. This must happen before JAX initializes
# its backend (for example, before calling jax.devices()).
HOST_DEVICE_COUNT = 4
numpyro.set_host_device_count(HOST_DEVICE_COUNT)

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_value

from rot_evol import (
    RotEvol,
    frame_ids_from_observation_log,
    load_isochrone_posterior_means,
    save_figures,
    save_inference_data,
    save_isochrone_mean_triangle,
    save_logage_histogram,
    save_logage_posterior_histogram,
    selection_label,
)
from rot_evol.data import ANALYSIS_ROOT, ISO_LABEL, find_frame_dir, result_path


jax.config.update("jax_enable_x64", True)

NUM_WARMUP = 4000
NUM_SAMPLES = 4000
NUM_CHAINS = HOST_DEVICE_COUNT
CHAIN_METHOD = "parallel"
TARGET_ACCEPT_PROB = 0.95
MAX_TREE_DEPTH = 12
RNG_SEED = 0

# The five global sampled parameters form a small correlated block. Per-star
# vector sites remain diagonal: densifying those sites would create a matrix
# of up to (5 * N_star)^2 elements and learn unnecessary cross-star terms.
DENSE_MASS = [
    (
        "a",
        "logb",
        "logvf",
        "lognorm_age_mu",
        "loglognorm_age_sigma",
    )
]

MASS_RANGES = ((0.9, 1.1), (1.1, 1.3))
FEH_RANGES = ((-0.1, None), (None, -0.1))
ROTATION_LAW = "power"
PREDICT_AGE_DIST = True
PREDICT_P_ROT = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate figures from completed mcmc.nc files without running MCMC",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun and replace completed mcmc.nc files",
    )
    parser.add_argument(
        "--selection-index",
        type=int,
        choices=range(1, 5),
        help="process only one mass/metallicity selection (1 through 4)",
    )
    return parser.parse_args()


def previous_isochrone_medians(frame_ids):
    """Use successful per-star fits as in-support joint-model starts."""
    values = {name: [] for name in ("logage", "feh_init", "eep", "distance")}
    for frame_id in frame_ids:
        frame_id = str(frame_id)
        nc_path = result_path(
            find_frame_dir(Path(ANALYSIS_ROOT), frame_id), ISO_LABEL
        )
        posterior = az.from_netcdf(nc_path).posterior
        for name in values:
            if name not in posterior:
                raise KeyError(f"{name!r} is absent from {nc_path}")
            values[name].append(float(np.median(np.asarray(posterior[name]))))
    return {name: np.asarray(items) for name, items in values.items()}


def result_label(mass_range, feh_range):
    return (
        f"isochrone_fit_{ROTATION_LAW}"
        f"_predict_age_{PREDICT_AGE_DIST}_Prot_{PREDICT_P_ROT}_"
        f"{selection_label(mass_range, feh_range)}"
    )


def run_mcmc(population, run_index):
    init_values = previous_isochrone_medians(population.frame_id)
    age_sigma = max(float(np.std(init_values["logage"])), 0.05)
    init_values.update(
        {
            "a": -0.5,
            "logb": 0.5,
            # Avoid the open-interval transform at the lower Uniform boundary.
            "logvf": 0.1,
            "lognorm_age_mu": float(np.mean(init_values["logage"])),
            "loglognorm_age_sigma": float(np.log10(age_sigma)),
            "cosi": np.full(population.N, 0.5),
        }
    )
    kernel = NUTS(
        population.isochrone_fit_model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=TARGET_ACCEPT_PROB,
        dense_mass=DENSE_MASS,
        max_tree_depth=MAX_TREE_DEPTH,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        chain_method=CHAIN_METHOD,
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(RNG_SEED + run_index),
        predict_age_dist=PREDICT_AGE_DIST,
        predict_P_rot=PREDICT_P_ROT,
        rotation_law=ROTATION_LAW,
    )
    mcmc.print_summary()
    return az.from_numpyro(mcmc)


def add_metadata(idata, population, mass_range, feh_range):
    idata.attrs.update(
        {
            "inference_model": "isochrone_fit",
            "rotation_law": ROTATION_LAW,
            "predict_age_dist": str(PREDICT_AGE_DIST),
            "predict_P_rot": str(PREDICT_P_ROT),
            "jax_backend": jax.default_backend(),
            "chain_method": CHAIN_METHOD,
            "dense_mass_blocks": repr(DENSE_MASS),
            "num_warmup": NUM_WARMUP,
            "num_samples": NUM_SAMPLES,
            "num_chains": NUM_CHAINS,
            "mass_range": str(mass_range),
            "feh_range": str(feh_range),
            "number_of_stars": population.N,
        }
    )


def save_plots(population, idata, label):
    output_dir = Path("result") / label
    means = load_isochrone_posterior_means(population.frame_id)
    paths = {
        "isochrone_triangle": save_isochrone_mean_triangle(
            means, output_dir / "isochrone_mass_age_feh_triangle.png"
        ),
        **save_figures(
            population,
            idata,
            label,
            predict_P_rot=PREDICT_P_ROT,
            rotation_law=ROTATION_LAW,
        ),
        "logage_hist": save_logage_histogram(
            population, output_dir / "logage_hist.png", idata=idata
        ),
        "logage_post_hist": save_logage_posterior_histogram(
            population, idata, output_dir / "logage_post_hist.png"
        ),
    }
    for name, path in paths.items():
        print(f"Saved figure ({name}): {path}", flush=True)


def main():
    args = parse_args()
    if args.plot_only and args.overwrite:
        raise ValueError("--plot-only and --overwrite cannot be used together")

    frame_ids = frame_ids_from_observation_log(
        "observation_log.csv",
        exclude_tags=("Trash",),
        keep_highest_sn_per_object=True,
        sn_column="Count (e-)",
    )
    all_population = RotEvol.from_analysis_results(
        frame_ids, refresh_cache=False, progress=True
    )
    selections = tuple(product(MASS_RANGES, FEH_RANGES))
    indexed_selections = list(enumerate(selections))
    if args.selection_index is not None:
        indexed_selections = [indexed_selections[args.selection_index - 1]]
    print(f"JAX backend/devices: {jax.default_backend()} / {jax.devices()}", flush=True)
    print(f"Chain method: {CHAIN_METHOD}", flush=True)
    print(f"Dense mass blocks: {DENSE_MASS}", flush=True)

    for run_index, (mass_range, feh_range) in indexed_selections:
        population = all_population.select(
            mass_range=mass_range, feh_range=feh_range
        )
        label = result_label(mass_range, feh_range)
        output_path = Path("result") / label / "mcmc.nc"
        print(
            f"\n[{run_index + 1}/{len(selections)}] Result label: {label}\n"
            f"{population.N} stars: {population.frame_id.tolist()}\n"
            f"mass={population.mass.min():.6f}..{population.mass.max():.6f}, "
            f"feh={population.feh.min():.6f}..{population.feh.max():.6f}",
            flush=True,
        )

        if output_path.is_file() and not args.overwrite:
            idata = az.from_netcdf(output_path)
            print(f"Reusing completed result: {output_path}", flush=True)
        elif args.plot_only:
            print(f"Skipping missing result in --plot-only mode: {output_path}", flush=True)
            continue
        else:
            idata = run_mcmc(population, run_index)
            add_metadata(idata, population, mass_range, feh_range)
            output_path = save_inference_data(population, idata, label)
            print(f"Saved inference data: {output_path}", flush=True)

        save_plots(population, idata, label)
        print(f"[{run_index + 1}/{len(selections)}] Completed: {label}", flush=True)


if __name__ == "__main__":
    main()
