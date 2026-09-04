#!/usr/bin/env python
# coding: utf-8

# # Rotation evolution
# 
# 実装は `rot_evol` パッケージに分割し、このスクリプトでは設定と実行のみを行う。結果は `result/<LABEL>/` に保存される。

# In[1]:


from itertools import product
from pathlib import Path
import arviz as az
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS
jax.config.update("jax_enable_x64", True)
print(jax.devices())
print(jax.device_count())
from rot_evol import (
    RESULT_ROOT,
    IsochronePosterior,
    RotEvol,
    frame_ids_from_observation_log,
    load_isochrone_posterior_means,
    normalize_rotation_law,
    save_figures,
    save_inference_data,
    save_isochrone_mean_triangle,
    save_logage_histogram,
    selection_label,
)


# In[2]:


ISO_NSAMPLES = 10000
NUM_WARMUP = 1000
NUM_SAMPLES = 1000
NUM_CHAINS = 4
# NumPyro suppresses per-chain NUTS diagnostics for vectorized chains.
# Sequential chains show step size and acceptance probability in the terminal.
CHAIN_METHOD = "vectorized"
TARGET_ACCEPT_PROB = 0.8
MAX_TREE_DEPTH = 12
RNG_SEED = 0
INFERENCE_MODEL = "standard"  # "importance" or "standard"
ROTATION_LAW = "power"  # "power", "broken_power", "gp", or "jaxspin"
BREAK_AGE_BOUNDS = (3, 6)  # Gyr; used by broken_power
RO_WMB_FACTOR_BOUNDS = (0.1, 2.0)  # inferred by jaxspin
JAXSPIN_FEH_BOUNDS = (-0.5, 0.5)  # inferred shared [Fe/H]
PREDICT_AGE_DIST = True
PREDICT_P_ROT = True
OBSERVATION_LOG = Path("observation_log.csv")
LOG_QUERY = None
MASS_RANGES = ((0.9, 1.1), (1.1, 1.3))
FEH_RANGES = ((None, -0.1), (-0.1, None))
RADIUS_RANGES = (None,)  # e.g. ((0.9, 1.1), (1.1, None)) [Rsun]
# MASS_RANGES = ((0.9, 1.1),)
# MASS_RANGES = ((1.1, 1.3),)
# FEH_RANGES = ((-0.1, None),)
# RADIUS_RANGES = ((0.9, 1.1),)

INFERENCE_MODEL = INFERENCE_MODEL.lower()
if INFERENCE_MODEL not in {"importance", "standard"}:
    raise ValueError("INFERENCE_MODEL must be importance or standard")
ROTATION_LAW = normalize_rotation_law(ROTATION_LAW)
if ROTATION_LAW == "jaxspin" and not PREDICT_P_ROT:
    raise ValueError("ROTATION_LAW=jaxspin requires PREDICT_P_ROT=True")

if ROTATION_LAW == "power":
    rotation_sites = ["a", "logb", "logvf"]
elif ROTATION_LAW == "broken_power":
    rotation_sites = ["a1", "break_age", "logb"]
elif ROTATION_LAW == "gp":
    rotation_sites = ["gp_latent", "log_gp_amplitude", "log_gp_scale", "logb"]
else:
    rotation_sites = ["Ro_wmb_factor", "feh"]
age_distribution_sites = (
    ["lognorm_age_mu", "loglognorm_age_sigma"]
    if PREDICT_AGE_DIST else []
)
dense_mass = [tuple([*rotation_sites, *age_distribution_sites])]

print(f"Inference model: {INFERENCE_MODEL}")
print(f"Rotation law: {ROTATION_LAW}")
print(f"JAX backend/devices: {jax.default_backend()} / {jax.devices()}")
print(f"Chain method: {CHAIN_METHOD}")
print(f"Dense mass block: {dense_mass[0]}")

frame_ids = frame_ids_from_observation_log(
    OBSERVATION_LOG,
    query=LOG_QUERY,
    exclude_tags=("Trash",),
    keep_highest_sn_per_object=True,
    sn_column="Count (e-)",
)
all_re = RotEvol.from_analysis_results(frame_ids, refresh_cache=False)

selections = tuple(product(MASS_RANGES, FEH_RANGES, RADIUS_RANGES))

for run_index, (mass_range, feh_range, radius_range) in enumerate(
    selections, start=1
):
    base_label = selection_label(mass_range, feh_range, radius_range)
    label = (
        f"{INFERENCE_MODEL}_{ROTATION_LAW}"
        f"_predict_age_{PREDICT_AGE_DIST}_Prot_{PREDICT_P_ROT}_{base_label}"
    )
    print(f"\n[{run_index}/{len(selections)}] Result label: {label}", flush=True)

    re = all_re.select(
        mass_range=mass_range,
        feh_range=feh_range,
        radius_range=radius_range,
    )
    isochrone_posterior = None
    if INFERENCE_MODEL == "importance" or PREDICT_P_ROT:
        isochrone_posterior = IsochronePosterior.from_analysis_results(
            re.frame_id,
            nsamples=ISO_NSAMPLES,
            refresh_cache=False,
        )
    print(f"{re.N} stars: {re.frame_id.tolist()}", flush=True)
    isochrone_means = load_isochrone_posterior_means(re.frame_id)
    triangle_path = save_isochrone_mean_triangle(
        isochrone_means,
        RESULT_ROOT / label / "isochrone_mass_age_feh_triangle.png",
    )
    print(f"Saved isochrone-mean triangle: {triangle_path}", flush=True)

    model_kwargs = {
        "predict_age_dist": PREDICT_AGE_DIST,
        "predict_P_rot": PREDICT_P_ROT,
        "rotation_law": ROTATION_LAW,
        "break_age_bounds": BREAK_AGE_BOUNDS,
        "ro_wmb_factor_bounds": RO_WMB_FACTOR_BOUNDS,
        "jaxspin_feh_bounds": JAXSPIN_FEH_BOUNDS,
    }
    if INFERENCE_MODEL == "importance":
        model_fn = re.importance_model
        model_args = (isochrone_posterior,)
    else:
        model_fn = re.model
        model_args = ()

    kernel = NUTS(
        model_fn,
        target_accept_prob=TARGET_ACCEPT_PROB,
        dense_mass=dense_mass,
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
        jax.random.PRNGKey(RNG_SEED + run_index - 1),
        *model_args,
        **model_kwargs,
    )

    idata = az.from_numpyro(mcmc)
    idata.attrs.update({
        "inference_model": INFERENCE_MODEL,
        "rotation_law": ROTATION_LAW,
        "predict_age_dist": str(PREDICT_AGE_DIST),
        "predict_P_rot": str(PREDICT_P_ROT),
        "jax_backend": jax.default_backend(),
        "chain_method": CHAIN_METHOD,
        "break_age_prior_lower_gyr": BREAK_AGE_BOUNDS[0],
        "break_age_prior_upper_gyr": BREAK_AGE_BOUNDS[1],
        "ro_wmb_factor_prior_lower": RO_WMB_FACTOR_BOUNDS[0],
        "ro_wmb_factor_prior_upper": RO_WMB_FACTOR_BOUNDS[1],
        "jaxspin_feh_prior_lower": JAXSPIN_FEH_BOUNDS[0],
        "jaxspin_feh_prior_upper": JAXSPIN_FEH_BOUNDS[1],
    })
    if ROTATION_LAW == "jaxspin":
        idata.attrs["jaxspin_fixed_mass_msun"] = float(np.median(re.mass))
    save_inference_data(re, idata, label)
    save_figures(
        re,
        idata,
        label,
        isochrone_posterior=isochrone_posterior,
        predict_P_rot=PREDICT_P_ROT,
        rotation_law=ROTATION_LAW,
    )
    save_logage_histogram(
        re,
        RESULT_ROOT / label / "logage_hist.png",
        idata=idata,
    )
    print(f"[{run_index}/{len(selections)}] Completed: {label}", flush=True)
