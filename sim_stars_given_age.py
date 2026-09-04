"""Generate mock isochrone posteriors for stars with known ages."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxstar
import numpy as np
import pandas as pd
from jax import random
from tqdm.auto import tqdm

CACHE_SCHEMA_VERSION = 1


def _mock_cache_path(
    cache_dir,
    *,
    logages_true,
    masses,
    feh_init,
    settings,
):
    """Build a stable cache path from every mock-generation input."""
    digest = hashlib.sha256()
    metadata = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "jaxstar_version": getattr(jaxstar, "__version__", "unknown"),
        **settings,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values in (
        ("logages_true", logages_true),
        ("masses", masses),
        ("feh_init", feh_init),
    ):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return Path(cache_dir) / f"isochrone_mock_{digest.hexdigest()[:20]}.npz"


def _load_mock_cache(cache_path, expected_stars):
    with np.load(cache_path, allow_pickle=False) as cached:
        required = {"cache_schema_version", "columns", "values"}
        missing = required.difference(cached.files)
        if missing:
            raise ValueError(f"Mock cache is missing arrays: {sorted(missing)}")
        version = int(cached["cache_schema_version"].item())
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported mock cache schema {version}; "
                f"expected {CACHE_SCHEMA_VERSION}"
            )
        columns = np.asarray(cached["columns"], dtype=str)
        values = np.asarray(cached["values"], dtype=float)
    if columns.ndim != 1 or columns.size == 0:
        raise ValueError("Mock cache columns must be a non-empty vector")
    if values.ndim != 3 or values.shape[0] != expected_stars:
        raise ValueError(
            f"Mock cache values must have shape (N_stars, N_draws, N_columns); "
            f"got {values.shape}"
        )
    if values.shape[2] != columns.size or not np.all(np.isfinite(values)):
        raise ValueError("Mock cache columns/values are inconsistent or non-finite")
    return [pd.DataFrame(star_values, columns=columns) for star_values in values]


def _save_mock_cache(cache_path, isochrone_posterior):
    if not isochrone_posterior:
        raise ValueError("Cannot cache an empty isochrone posterior")
    columns = np.asarray(isochrone_posterior[0].columns, dtype=str)
    values = []
    for index, samples in enumerate(isochrone_posterior):
        if list(samples.columns) != columns.tolist():
            raise ValueError(f"Posterior columns differ for star {index}")
        values.append(samples.to_numpy(dtype=float))
    sample_shapes = {value.shape for value in values}
    if len(sample_shapes) != 1:
        raise ValueError("All mock posteriors must have the same shape")
    values = np.stack(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Mock posterior cache values must be finite")

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION),
            columns=columns,
            values=values,
        )
    os.replace(temporary_path, cache_path)


def simulate_stars_given_age(
    logages_true: Sequence[float],
    *,
    masses: float | Sequence[float] = 1.0,
    feh_init: float | Sequence[float] = 0.0,
    teff_err: float = 100.0,
    gmag_err: float = 0.02,
    feh_err: float = 0.1,
    parallax: float = 100.0,
    parallax_err: float = 0.01,
    rng_seed: int = 0,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    target_accept_prob: float = 0.99,
    max_tree_depth: int = 15,
    cache_dir: str | Path = Path("sim/cache"),
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[pd.DataFrame]:
    """Simulate stellar observables and infer one isochrone posterior per star.

    Parameters
    ----------
    logages_true
        True stellar ages as ``log10(age / yr)``.
    masses
        True stellar masses in solar units. A scalar is broadcast to all stars.
    feh_init
        Initial metallicities in dex. A scalar is broadcast to all stars.
    num_samples
        Number of retained HMC draws per chain.
    cache_dir
        Directory for compressed mock-posterior caches.
    use_cache
        Load and save a cache keyed by all generation and MCMC inputs.
    refresh_cache
        Ignore an existing matching cache and regenerate it.

    Returns
    -------
    list of pandas.DataFrame
        ``isochrone_posterior[i]`` contains the posterior draws for star ``i``. In
        particular, ``isochrone_posterior[i]["age"]`` is in Gyr and
        ``isochrone_posterior[i]["radius"]`` is in solar radii.
    """
    logages_true = np.asarray(logages_true, dtype=float)
    if logages_true.ndim != 1 or logages_true.size == 0:
        raise ValueError("logages_true must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(logages_true)):
        raise ValueError("logages_true must contain only finite values")

    def broadcast_parameter(value, name):
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = np.full(logages_true.size, array.item())
        if array.shape != logages_true.shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite and scalar or match logages_true")
        return array

    masses = broadcast_parameter(masses, "masses")
    feh_init = broadcast_parameter(feh_init, "feh_init")
    if np.any(masses <= 0):
        raise ValueError("masses must be positive")

    errors = np.asarray([teff_err, gmag_err, feh_err, parallax_err], dtype=float)
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError("all observational errors must be finite and positive")
    if not np.isfinite(parallax) or parallax <= 0:
        raise ValueError("parallax must be finite and positive")
    if not isinstance(use_cache, bool) or not isinstance(refresh_cache, bool):
        raise ValueError("use_cache and refresh_cache must be booleans")
    for value, name in (
        (num_warmup, "num_warmup"),
        (num_samples, "num_samples"),
        (num_chains, "num_chains"),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    cache_path = _mock_cache_path(
        cache_dir,
        logages_true=logages_true,
        masses=masses,
        feh_init=feh_init,
        settings={
            "teff_err": float(teff_err),
            "gmag_err": float(gmag_err),
            "feh_err": float(feh_err),
            "parallax": float(parallax),
            "parallax_err": float(parallax_err),
            "rng_seed": int(rng_seed),
            "num_warmup": int(num_warmup),
            "num_samples": int(num_samples),
            "num_chains": int(num_chains),
            "target_accept_prob": float(target_accept_prob),
            "max_tree_depth": int(max_tree_depth),
            "linear_age": True,
            "flat_age_marginal": False,
            "nodata": False,
        },
    )
    if use_cache and cache_path.is_file() and not refresh_cache:
        cached = _load_mock_cache(cache_path, logages_true.size)
        print(f"Loaded mock isochrone cache: {cache_path}")
        return cached

    mf = jaxstar.mistfit.MistFit()
    mf.mg.set_keys(["teff", "gmag3", "feh_photosphere"])
    values = jax.vmap(mf.mg.values_given_mass, in_axes=(0, 0, 0))(
        jnp.asarray(logages_true),
        jnp.asarray(feh_init),
        jnp.asarray(masses),
    )
    teff_true, gmag_true, feh_true = (np.asarray(value) for value in values)

    rng = np.random.default_rng(rng_seed)
    observations = np.column_stack(
        (
            teff_true + rng.normal(0.0, teff_err, logages_true.size),
            gmag_true + rng.normal(0.0, gmag_err, logages_true.size),
            feh_true + rng.normal(0.0, feh_err, logages_true.size),
            parallax + rng.normal(0.0, parallax_err, logages_true.size),
        )
    )

    keys = random.split(random.PRNGKey(rng_seed), logages_true.size)
    isochrone_posterior = []
    stars = zip(observations, keys, strict=True)
    for observation, rng_key in tqdm(
        stars,
        total=logages_true.size,
        desc="Isochrone fits",
        unit="star",
    ):
        mf.set_data(
            ["teff", "gmag3", "feh", "parallax"],
            observation,
            errors,
        )
        mf.setup_hmc(
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )
        mf.mcmc.progress_bar = False
        mf.mcmc.run(
            rng_key,
            linear_age=True,
            flat_age_marginal=False,
            nodata=False,
        )
        samples = pd.DataFrame(mf.mcmc.get_samples())
        isochrone_posterior.append(samples)

    if use_cache:
        _save_mock_cache(cache_path, isochrone_posterior)
        print(f"Saved mock isochrone cache: {cache_path}")
    return isochrone_posterior


def stack_isochrone_samples(
    isochrone_posterior: Sequence[pd.DataFrame],
    *,
    nsamples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the age/radius columns of ``isochrone_posterior`` into ``(N, K)`` arrays."""
    if not isochrone_posterior:
        raise ValueError("isochrone_posterior must contain at least one posterior")

    ages = []
    radii = []
    for index, samples in enumerate(isochrone_posterior):
        missing = {"age", "radius"}.difference(samples.columns)
        if missing:
            raise ValueError(f"isochrone_posterior[{index}] is missing columns: {sorted(missing)}")
        age = np.asarray(samples["age"], dtype=float)
        radius = np.asarray(samples["radius"], dtype=float)
        if nsamples is not None:
            if not isinstance(nsamples, int) or nsamples <= 0:
                raise ValueError("nsamples must be a positive integer or None")
            if age.size < nsamples:
                raise ValueError(
                    f"isochrone_posterior[{index}] has {age.size} draws; requested {nsamples}"
                )
            age = age[:nsamples]
            radius = radius[:nsamples]
        ages.append(age)
        radii.append(radius)

    sample_counts = {age.size for age in ages}
    if len(sample_counts) != 1:
        raise ValueError("all posteriors in isochrone_posterior must have the same number of draws")

    age = np.stack(ages)
    radius = np.stack(radii)
    if not np.all(np.isfinite(age)) or not np.all(np.isfinite(radius)):
        raise ValueError("isochrone_posterior age/radius samples must be finite")
    if np.any(age <= 0) or np.any(radius <= 0):
        raise ValueError("isochrone_posterior age/radius samples must be positive")
    return age, radius
