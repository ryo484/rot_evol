#!/usr/bin/env python
# coding: utf-8

# # Rotation-law parameter recovery simulation
#
# 既知のパラメータから模擬観測を生成し、`re_sim.importance_model` または
# `re_sim.model` で回転則のパラメータを回収できるか確認する。回転則は単一の
# power law と、指定年齢で連続な broken power law から選択できる。

# In[1]:


from datetime import datetime
from pathlib import Path

import arviz as az
import corner
import jax
import matplotlib.pyplot as plt
import numpy as np
from numpyro.infer import MCMC, NUTS
from scipy.stats import truncnorm

from rot_evol import *
from sim_stars_given_age import simulate_stars_given_age, stack_isochrone_samples

jax.config.update("jax_enable_x64", True)
print(f"JAX backend/devices: {jax.default_backend()} / {jax.devices()}")


# ## Generate synthetic observations
# 
# 年齢は対数的に広く配置し、自転軸は等方的（$\cos i\sim\mathrm{Uniform}(0,1)$）とする。$v\sin i$ にガウス誤差を加える。
# `simulate_stars_given_age` で測光・分光量を模擬して星ごとの isochrone posterior を推定し、その年齢分布を `IsochronePosterior` に入力する。

# In[2]:


RNG_SEED = 1
N_STARS = 50

# Change only these two values to select the inference model and rotation law.
INFERENCE_MODEL = "standard"  # "importance" or "standard"
ROTATION_LAW = "power"  # "power", "broken_power", "gp", or "jaxspin"
BREAK_AGE_TRUE = 4.6  # Gyr; mock-data truth for broken_power
BREAK_AGE_BOUNDS = (3, 6)  # Gyr; Uniform prior bounds
ISO_NUM_CHAINS = 4
ISO_NUM_WARMUP = 1000
ISO_NUM_SAMPLES_PER_CHAIN = 1000
N_ISO_SAMPLES = ISO_NUM_CHAINS * ISO_NUM_SAMPLES_PER_CHAIN
A_TRUE = -0.5
A1_TRUE = -0.5
A2_TRUE = -0.0
B_TRUE = 25  # day if PREDICT_P_ROT, otherwise km/s
VSINI_ERR = 1  # km/s
TEFF_ERR = 100  # K
VF_TRUE = 5  # day if PREDICT_P_ROT, otherwise km/s
GMAG_ERR = 0.02  # mag
FEH_ERR = 0.1  # dex
PARALLAX = 100  # mas
PARALLAX_ERR = 0.01  # mas

LOGAGE_PRIOR_MU_TRUE = 9.6  # log10(age/yr)
LOGAGE_PRIOR_SIGMA_TRUE = 0.3

NUM_WARMUP = 4000
NUM_SAMPLES = 4000
NUM_CHAINS = 4
CHAIN_METHOD = "vectorized"  # suitable for multiple chains on one GPU
TARGET_ACCEPT_PROB = 0.80
MAX_TREE_DEPTH = 8
PREDICT_P_ROT = True

INFERENCE_MODEL = INFERENCE_MODEL.lower()
if INFERENCE_MODEL not in {"importance", "standard"}:
    raise ValueError("INFERENCE_MODEL must be importance or standard")
ROTATION_LAW = normalize_rotation_law(ROTATION_LAW)

if ROTATION_LAW == "power":
    slope = -A_TRUE if PREDICT_P_ROT else A_TRUE
    TRUE_SLOPES = (slope, VF_TRUE)
    ROTATION_TRUTH = {"a": slope, "b": B_TRUE, "vf": VF_TRUE}
elif ROTATION_LAW == "broken_power":
    slopes = (
        -A1_TRUE if PREDICT_P_ROT else A1_TRUE,
        -A2_TRUE if PREDICT_P_ROT else A2_TRUE,
    )
    TRUE_SLOPES = slopes
    ROTATION_TRUTH = {
        "a1": slopes[0],
        "a2": slopes[1],
        "b": B_TRUE,
        "break_age": BREAK_AGE_TRUE,
    }
else:
    raise NotImplementedError(
        f"ROTATION_LAW={ROTATION_LAW} is reserved for a future implementation"
    )
ROTATION_PARAMETER_NAMES = tuple(ROTATION_TRUTH)
AGE_DISTRIBUTION_PARAMETER_NAMES = (
    "lognorm_age_mu",
    "lognorm_age_sigma",
)
SUMMARY_PARAMETER_NAMES = (
    *ROTATION_PARAMETER_NAMES,
    *AGE_DISTRIBUTION_PARAMETER_NAMES,
)


def rotation_relation(age, slopes=TRUE_SLOPES, b=B_TRUE):
    return evaluate_rotation_law(
        age, ROTATION_LAW, slopes, b, BREAK_AGE_TRUE
    )


def label_number(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")

SIM_ROOT = Path("sim")
RUN_TIMESTAMP = datetime.now().strftime("%y%m%d%H")
rotation_label = "_".join(
    f"{name}true{label_number(value)}" for name, value in ROTATION_TRUTH.items()
)
SIM_LABEL = (
    f"{RUN_TIMESTAMP}_{INFERENCE_MODEL}_{ROTATION_LAW}"
    f"_Nstar{N_STARS}_{rotation_label}"
    f"_tefferr{label_number(TEFF_ERR)}_gmagerr{label_number(GMAG_ERR)}"
    f"_feherr{label_number(FEH_ERR)}_plxerr{label_number(PARALLAX_ERR)}"
    f"_vsinierr{label_number(VSINI_ERR)}"
    f"_logage_mu{label_number(LOGAGE_PRIOR_MU_TRUE)}"
    f"_logage_sigma{label_number(LOGAGE_PRIOR_SIGMA_TRUE)}"
)
SIM_OUTPUT_DIR = SIM_ROOT / SIM_LABEL
SIM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RNG_SEED)
LOGAGE_MIN = 9.0
LOGAGE_MAX = np.log10(11e9)
lower = (LOGAGE_MIN - LOGAGE_PRIOR_MU_TRUE) / LOGAGE_PRIOR_SIGMA_TRUE
upper = (LOGAGE_MAX - LOGAGE_PRIOR_MU_TRUE) / LOGAGE_PRIOR_SIGMA_TRUE
logage_true = truncnorm.rvs(
    lower, upper, loc=LOGAGE_PRIOR_MU_TRUE, scale=LOGAGE_PRIOR_SIGMA_TRUE,
    size=N_STARS, random_state=RNG_SEED,
)
age_true = 10 ** (logage_true - 9.0)  # Gyr

isochrone_posterior = simulate_stars_given_age(
    logage_true,
    teff_err=TEFF_ERR, gmag_err=GMAG_ERR, feh_err=FEH_ERR,
    parallax=PARALLAX, parallax_err=PARALLAX_ERR,
    rng_seed=RNG_SEED, num_warmup=ISO_NUM_WARMUP,
    num_samples=ISO_NUM_SAMPLES_PER_CHAIN, num_chains=ISO_NUM_CHAINS,
)
age_samples, radius_samples = stack_isochrone_samples(isochrone_posterior)
age_obs = np.median(age_samples, axis=1)
age_err = np.std(age_samples, axis=1, ddof=1)

cosi_true = rng.uniform(size=N_STARS)
rotation_true = np.asarray(rotation_relation(age_true))
radius_median = np.median(radius_samples, axis=1)
radius_std = np.std(radius_samples, axis=1)
if PREDICT_P_ROT:
    v_true = 2.0 * np.pi * radius_median * R_SUN_KM / (rotation_true * DAY_S)
else:
    v_true = rotation_true
vsini_true = v_true * np.sqrt(1.0 - cosi_true**2)
vsini_obs = np.abs(vsini_true + rng.normal(0.0, VSINI_ERR, N_STARS))

re_sim = RotEvol(
    frame_id=[f"sim-{i:03d}" for i in range(N_STARS)],
    vsini=vsini_obs, vsini_err=np.full(N_STARS, VSINI_ERR),
    age=age_obs, age_err=age_err,
    mass=np.ones(N_STARS), mass_err=np.full(N_STARS, 0.05),
    feh=np.zeros(N_STARS), feh_err=np.full(N_STARS, FEH_ERR),
    radius=radius_median, radius_err=np.maximum(radius_std, 1.0e-12),
)
iso_sim = IsochronePosterior(
    frame_id=re_sim.frame_id, age=age_samples, radius=radius_samples,
)

print(f"Output directory: {SIM_OUTPUT_DIR}")
print(f"Generated {re_sim.N} stars with {iso_sim.Nsamples} isochrone samples each")
print(f"Inference model: {INFERENCE_MODEL}")
print(f"Rotation law: {ROTATION_LAW}")
print("True parameters:", ROTATION_TRUTH)


# In[3]:


age_grid = np.geomspace(0.1, 11.0, 300)
fig, ax = plt.subplots(figsize=(7, 5))
age_q16, age_q50, age_q84 = np.quantile(age_samples, [0.16, 0.50, 0.84], axis=1)
age_xerr = np.vstack((age_q50 - age_q16, age_q84 - age_q50))


def identity_plot_limits(*arrays):
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("No positive finite ages are available for log axes")
    log_limits = np.log10([positive.min(), positive.max()])
    log_span = max(log_limits[1] - log_limits[0], 0.1)
    return 10 ** np.array([
        log_limits[0] - 0.05 * log_span,
        log_limits[1] + 0.05 * log_span,
    ])


age_observation_limits = identity_plot_limits(age_true, age_q16, age_q84)
age_fig, age_ax = plt.subplots(figsize=(5, 5))
age_ax.errorbar(
    age_true,
    age_q50,
    yerr=age_xerr,
    fmt="o",
    ms=4,
    color="C0",
    ecolor="0.65",
    capsize=2,
)
age_ax.plot(age_observation_limits, age_observation_limits, color="black", ls="--", label="1:1")
age_ax.set(
    xscale="log",
    yscale="log",
    xlim=age_observation_limits,
    ylim=age_observation_limits,
    xlabel="True age [Gyr]",
    ylabel="Observed age [Gyr]",
)
age_ax.legend()
age_fig.tight_layout()
age_fig.savefig(
    SIM_OUTPUT_DIR / "age_true_vs_age_obs.png",
    dpi=200,
    bbox_inches="tight",
)

if PREDICT_P_ROT:
    observed_relation = 2.0 * np.pi * radius_median * R_SUN_KM / (vsini_obs * DAY_S)
    observed_relation_err = observed_relation * np.sqrt(
        (radius_std / radius_median) ** 2 + (VSINI_ERR / vsini_obs) ** 2
    )
    relation_label = r"mock $P/\sin i$"
    truth_label = "true rotation period"
    relation_ylabel = "Rotation period [day]"
else:
    observed_relation = vsini_obs
    observed_relation_err = np.full(N_STARS, VSINI_ERR)
    relation_label = r"mock $v\sin i$"
    truth_label = "true equatorial velocity"
    relation_ylabel = r"Velocity [km s$^{-1}$]"
ax.errorbar(
    age_q50,
    observed_relation,
    xerr=age_xerr,
    yerr=observed_relation_err,
    fmt="o",
    ms=4,
    alpha=0.6,
    label=relation_label,
)
ax.plot(age_grid, rotation_relation(age_grid), color="black", lw=2, label=truth_label)
ax.set(xlabel="Age [Gyr]", ylabel=relation_ylabel,
    #    ylim=[None,None],xlim=[None,None],
       xscale="log",yscale="log"
       )
ax.legend()
fig.tight_layout()
fig.savefig(SIM_OUTPUT_DIR / "mock_data.png", dpi=200, bbox_inches="tight")


# ## Fit with the selected inference model
#
# `INFERENCE_MODEL` で isochrone posterior を周辺化する `importance_model` と、
# 代表値・誤差を用いる `model` を切り替える。`ROTATION_LAW` は両方に共通する。

# In[ ]:


model_kwargs = {
    "predict_age_dist": True,
    "predict_P_rot": PREDICT_P_ROT,
    "rotation_law": ROTATION_LAW,
    "break_age_bounds": BREAK_AGE_BOUNDS,
}
if INFERENCE_MODEL == "importance":
    model_fn = re_sim.importance_model
    model_args = (iso_sim,)
else:
    model_fn = re_sim.model
    model_args = ()

kernel = NUTS(
    model_fn,
    target_accept_prob=TARGET_ACCEPT_PROB,
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
    jax.random.PRNGKey(RNG_SEED + 1),
    *model_args,
    **model_kwargs,
)
idata = az.from_numpyro(mcmc)
idata.attrs.update({
    "simulation_label": SIM_LABEL,
    "rng_seed": RNG_SEED,
    "n_stars": N_STARS,
    "n_iso_samples": N_ISO_SAMPLES,
    "inference_model": INFERENCE_MODEL,
    "rotation_law": ROTATION_LAW,
    "break_age_true_gyr": BREAK_AGE_TRUE,
    "break_age_prior_lower_gyr": BREAK_AGE_BOUNDS[0],
    "break_age_prior_upper_gyr": BREAK_AGE_BOUNDS[1],
    "age_error_source": "isochrone_posterior",
    "vsini_err_km_s": VSINI_ERR,
    "logage_mu_true": LOGAGE_PRIOR_MU_TRUE,
    "logage_sigma_true": LOGAGE_PRIOR_SIGMA_TRUE,
    "num_warmup": NUM_WARMUP,
    "num_samples": NUM_SAMPLES,
    "num_chains": NUM_CHAINS,
    "target_accept_prob": TARGET_ACCEPT_PROB,
    "max_tree_depth": MAX_TREE_DEPTH,
    "predict_P_rot": str(PREDICT_P_ROT),
    "jax_backend": jax.default_backend(),
    "chain_method": CHAIN_METHOD,
})
for parameter_name, truth in ROTATION_TRUTH.items():
    idata.attrs[f"{parameter_name}_true"] = truth
az.summary(
    idata,
    var_names=list(SUMMARY_PARAMETER_NAMES),
    hdi_prob=0.68,
)


# ## Save outputs
# 
# `run_rot_evol.py` と同じ保存関数を使い、推定結果と標準診断図を `sim/<SIM_LABEL>/` に保存する。

# In[ ]:


inference_path = save_inference_data(
    re_sim,
    idata,
    SIM_LABEL,
    result_root=SIM_ROOT,
)
standard_figure_paths = save_figures(
    re_sim,
    idata,
    SIM_LABEL,
    isochrone_posterior=iso_sim,
    predict_P_rot=PREDICT_P_ROT,
    rotation_law=ROTATION_LAW,
    result_root=SIM_ROOT,
)
logage_histogram_path = save_logage_histogram(
    re_sim,
    SIM_OUTPUT_DIR / "logage_hist.png",
    idata=idata,
)

saved_paths = {
    "inference": inference_path,
    **standard_figure_paths,
    "logage_hist": logage_histogram_path,
    "mock_data": SIM_OUTPUT_DIR / "mock_data.png",
    "age_true_vs_age_obs": SIM_OUTPUT_DIR / "age_true_vs_age_obs.png",
}
print(f"Saved standard outputs under {SIM_OUTPUT_DIR}")
for name, saved_path in saved_paths.items():
    print(f"  {name}: {saved_path}")
print("Additional simulation figures are saved by the recovery cells below:")
for filename in ("recovery_trace.png", "recovery_corner.png", "recovery_relation.png"):
    print(f"  {SIM_OUTPUT_DIR / filename}")


# ## Recovery check
# 
# 事後中央値と68% HDIを真値と比較する。有限個の模擬データなので中央値が真値と完全一致する必要はないが、正しく較正された推定では真値が信用区間内に入ることが期待される。

# In[ ]:


posterior = idata.posterior.stack(sample=("chain", "draw"))

def recovery_row(name, truth):
    values = np.asarray(posterior[name])
    low, median, high = np.quantile(values, [0.03, 0.5, 0.97])
    return {
        "parameter": name,
        "truth": truth,
        "median": median,
        "94% interval": (low, high),
        "truth in interval": bool(low <= truth <= high),
    }

recovery = [
    recovery_row(name, truth) for name, truth in ROTATION_TRUTH.items()
]
for row in recovery:
    print(row)

all_recovered = all(row["truth in interval"] for row in recovery)
print("All rotation parameters recovered within their 94% intervals:", all_recovered)


# In[ ]:


trace_axes = az.plot_trace(
    idata,
    var_names=list(SUMMARY_PARAMETER_NAMES),
    compact=False,
)
trace_figure = np.asarray(trace_axes).flat[0].figure
trace_figure.tight_layout()
trace_figure.savefig(
    SIM_OUTPUT_DIR / "recovery_trace.png",
    dpi=200,
    bbox_inches="tight",
)


# In[ ]:


corner_samples = np.column_stack([
    np.asarray(posterior[name]) for name in SUMMARY_PARAMETER_NAMES
])
parameter_labels = {
    "a": r"$a$",
    "a1": r"$a_1$",
    "a2": r"$a_2$",
    "b": (r"$b$ [day]" if PREDICT_P_ROT else r"$b$ [km s$^{-1}$]"),
    "break_age": r"$t_{\rm break}$ [Gyr]",
    "vf": (r"$v_f$ [day]" if PREDICT_P_ROT else r"$v_f$ [km s$^{-1}$]"),
    "lognorm_age_mu": r"$\mu_{\log \tau}$",
    "lognorm_age_sigma": r"$\sigma_{\log \tau}$",
}
corner_truths = [
    *ROTATION_TRUTH.values(),
    LOGAGE_PRIOR_MU_TRUE,
    LOGAGE_PRIOR_SIGMA_TRUE,
]
figure = corner.corner(
    corner_samples,
    labels=[parameter_labels[name] for name in SUMMARY_PARAMETER_NAMES],
    truths=corner_truths,
    truth_color="C3",
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 11},
)
figure.suptitle(f"Recovery with {ROTATION_LAW}", y=1.02)
figure.savefig(SIM_OUTPUT_DIR / "recovery_corner.png", dpi=200, bbox_inches="tight")


# In[ ]:


v_draw = posterior_rotation_draws(
    age_grid[None, :], posterior, ROTATION_LAW
)
low, median, high = np.quantile(v_draw, [0.16, 0.5, 0.84], axis=0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    age_obs,
    observed_relation,
    xerr=age_err,
    yerr=observed_relation_err,
    fmt=".",
    ms=4,
    alpha=0.6,
    label=relation_label,
    elinewidth=0.5,
    color="gray",
)

ax.fill_between(age_grid, low, high, alpha=0.25, color="C0", label="68% posterior interval")
ax.plot(age_grid, median, color="C0", lw=2, label="posterior median")
ax.plot(age_grid, rotation_relation(age_grid), color="black", ls="--", lw=2, label="truth")
ax.set(
    xscale="log", yscale="log", xlabel="Age [Gyr]", ylabel=relation_ylabel,
    xlim=(1, 11), ylim=(0.1, 100)
       )
ax.legend()
fig.tight_layout()
fig.savefig(SIM_OUTPUT_DIR / "recovery_relation.png", dpi=200, bbox_inches="tight")

