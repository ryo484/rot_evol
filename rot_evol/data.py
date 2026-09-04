"""Data containers and analysis-result loaders."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .models import RotEvolModelMixin

ANALYSIS_ROOT = Path("/mnt6tb/data/analysis_results")
SPEC_LABEL = "orders03-08"
ISO_LABEL = "mistfit_g_st38"
RESULT_ROOT = Path("result")

def normalize_frame_id(value):
    value = str(value).strip().removeprefix("GRA").removeprefix("G")
    return value.zfill(8) if value.isdigit() else value

def posterior_mean_std(nc_path, variable):
    """Return posterior mean and standard deviation over chains and draws."""
    idata = az.from_netcdf(nc_path)
    if variable not in idata.posterior:
        raise KeyError(f"{variable!r} is absent from {nc_path}")
    values = np.asarray(idata.posterior[variable], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid posterior mean/std for {variable!r} in {nc_path}")
    return mean, std

def posterior_median_std(nc_path, variable):
    """Return posterior median and standard deviation over chains and draws."""
    idata = az.from_netcdf(nc_path)
    if variable not in idata.posterior:
        raise KeyError(f"{variable!r} is absent from {nc_path}")
    values = np.asarray(idata.posterior[variable], dtype=float)
    median = float(np.median(values))
    std = float(np.std(values))
    if not np.isfinite(median) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid posterior median/std for {variable!r} in {nc_path}")
    return median, std


def isochrone_observations_from_metadata(metadata_path):
    """Load extinction-corrected Gaia observables used by a mistfit run."""
    metadata_path = Path(metadata_path)
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    gaia = metadata.get("gaia_dr3", {})
    if not isinstance(gaia, dict):
        raise ValueError(f"Invalid gaia_dr3 metadata in {metadata_path}")
    keys = {
        "gmag3_mean": "gmag3_extinction_corrected",
        "gmag3_err": "gmag3_extinction_corrected_err",
        "parallax_mean": "gaia_parallax_mas",
        "parallax_err": "gaia_parallax_error_mas",
    }
    missing = [source for source in keys.values() if source not in gaia]
    if missing:
        raise KeyError(f"Missing isochrone metadata keys {missing} in {metadata_path}")
    values = {target: float(gaia[source]) for target, source in keys.items()}
    if not np.all(np.isfinite(list(values.values()))):
        raise ValueError(f"Non-finite isochrone observables in {metadata_path}")
    if values["gmag3_err"] <= 0 or values["parallax_err"] <= 0:
        raise ValueError(f"Non-positive isochrone observable error in {metadata_path}")
    return values


def find_frame_dir(root, frame_id):
    matches = sorted(p for p in root.rglob(frame_id) if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"frame_id={frame_id} was not found below {root}")
    if len(matches) > 1:
        detail = "\n".join(map(str, matches))
        raise ValueError(f"frame_id={frame_id} is not unique below {root}:\n{detail}")
    return matches[0]

def result_path(frame_dir, label):
    path = frame_dir / label / "mcmc.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Missing result: {path}")
    return path

def frame_ids_from_observation_log(
    path,
    query=None,
    exclude_tags=None,
    keep_highest_sn_per_object=False,
    sn_column="Count (e-)",
):
    """Read frame IDs from an observation log, preserving the selected order."""
    table = pd.read_csv(path)
    required = {"Frame"}
    if exclude_tags:
        required.add("Tags")
    if keep_highest_sn_per_object:
        required.update({"Object", sn_column})
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing observation-log columns: {sorted(missing)}")
    if query:
        table = table.query(query)
    if exclude_tags:
        tags = table["Tags"].fillna("").astype(str)
        for excluded in exclude_tags:
            tags = tags.mask(tags.str.contains(str(excluded), case=False, regex=False), "__EXCLUDED__")
        table = table.loc[tags != "__EXCLUDED__"]
    if keep_highest_sn_per_object:
        table = table.copy()
        table["_row_order"] = np.arange(len(table))
        table["_sn"] = pd.to_numeric(table[sn_column], errors="coerce").fillna(-np.inf)
        table = (
            table.sort_values(["_sn", "_row_order"], ascending=[False, True])
            .drop_duplicates("Object", keep="first")
            .sort_values("_row_order")
        )
    frame_ids = [normalize_frame_id(x) for x in table["Frame"].dropna()]
    return list(dict.fromkeys(frame_ids))


@dataclass(frozen=True)
class RotEvol(RotEvolModelMixin):
    frame_id: np.ndarray

    vsini: np.ndarray

    vsini_err: np.ndarray

    age: np.ndarray

    age_err: np.ndarray

    mass: np.ndarray

    mass_err: np.ndarray

    feh: np.ndarray

    feh_err: np.ndarray

    radius: np.ndarray

    radius_err: np.ndarray

    logage: np.ndarray | None = None

    logage_err: np.ndarray | None = None

    teff_mean: np.ndarray | None = None

    teff_err: np.ndarray | None = None

    feh_mean: np.ndarray | None = None

    gmag3_mean: np.ndarray | None = None

    gmag3_err: np.ndarray | None = None

    parallax_mean: np.ndarray | None = None

    parallax_err: np.ndarray | None = None

    def __post_init__(self):
        names = (
            "vsini", "vsini_err", "age", "age_err", "mass", "mass_err",
            "feh", "feh_err", "radius", "radius_err",
        )
        frame_id = np.asarray(self.frame_id, dtype=str)
        arrays = {name: np.asarray(getattr(self, name), dtype=float) for name in names}
        if frame_id.ndim != 1 or frame_id.size == 0:
            raise ValueError("frame_id must be a non-empty one-dimensional array")
        if len(set(frame_id)) != frame_id.size:
            raise ValueError("frame_id contains duplicates")
        if any(x.ndim != 1 or x.size != frame_id.size for x in arrays.values()):
            raise ValueError("All parameter arrays must be one-dimensional and match frame_id")
        if any(not np.all(np.isfinite(x)) for x in arrays.values()):
            raise ValueError("All parameter arrays must be finite")
        if any(np.any(arrays[name] <= 0) for name in ("vsini_err", "age_err", "mass_err", "feh_err", "radius_err")):
            raise ValueError("All standard deviations must be positive")
        if any(np.any(arrays[name] <= 0) for name in ("age", "mass", "radius")):
            raise ValueError("age, mass, and radius must be positive")
        if self.logage is None and self.logage_err is None:
            logage = np.log10(arrays["age"]) + 9.0
            logage_err = arrays["age_err"] / (arrays["age"] * np.log(10.0))
        elif self.logage is None or self.logage_err is None:
            raise ValueError("logage and logage_err must be provided together")
        else:
            logage = np.asarray(self.logage, dtype=float)
            logage_err = np.asarray(self.logage_err, dtype=float)
        if logage.ndim != 1 or logage.size != frame_id.size:
            raise ValueError("logage must be one-dimensional and match frame_id")
        if logage_err.ndim != 1 or logage_err.size != frame_id.size:
            raise ValueError("logage_err must be one-dimensional and match frame_id")
        if not np.all(np.isfinite(logage)) or not np.all(np.isfinite(logage_err)):
            raise ValueError("logage and logage_err must be finite")
        if np.any(logage_err <= 0):
            raise ValueError("logage_err must be positive")
        object.__setattr__(self, "frame_id", frame_id)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "logage", logage)
        object.__setattr__(self, "logage_err", logage_err)
        isochrone_names = (
            "teff_mean", "teff_err", "feh_mean", "gmag3_mean", "gmag3_err",
            "parallax_mean", "parallax_err",
        )
        supplied = [getattr(self, name) is not None for name in isochrone_names]
        if any(supplied) and not all(supplied):
            missing = [name for name in isochrone_names if getattr(self, name) is None]
            raise ValueError(
                "Isochrone observables must be provided together; "
                f"missing {missing}"
            )
        if all(supplied):
            isochrone_arrays = {
                name: np.asarray(getattr(self, name), dtype=float)
                for name in isochrone_names
            }
            if any(
                value.ndim != 1 or value.size != frame_id.size
                for value in isochrone_arrays.values()
            ):
                raise ValueError(
                    "Isochrone observable arrays must be one-dimensional and match frame_id"
                )
            if any(not np.all(np.isfinite(value)) for value in isochrone_arrays.values()):
                raise ValueError("Isochrone observable arrays must be finite")
            if any(
                np.any(isochrone_arrays[name] <= 0)
                for name in ("teff_err", "gmag3_err", "parallax_err")
            ):
                raise ValueError("Isochrone observable standard deviations must be positive")
            if np.any(isochrone_arrays["teff_mean"] <= 0):
                raise ValueError("teff_mean must be positive")
            for name, value in isochrone_arrays.items():
                object.__setattr__(self, name, value)

    @property
    def N(self):
        return self.frame_id.size

    @property
    def logage_prior_loc(self):
        """Shared logage-prior center from all selected stars."""
        return float(np.median(self.logage))

    @property
    def logage_prior_scale(self):
        """Shared logage-prior width from the selected stars' median ages."""
        scale = float(np.std(self.logage))
        return scale if scale > 0 else float(np.median(self.logage_err))

    def select(self, mass_range=None, feh_range=None, radius_range=None):
        """Return rows inside optional mass, [Fe/H], and radius ranges."""
        mask = np.ones(self.N, dtype=bool)
        for values, bounds, name in (
            (self.mass, mass_range, "mass"),
            (self.feh, feh_range, "feh"),
            (self.radius, radius_range, "radius"),
        ):
            if bounds is None:
                continue
            if len(bounds) != 2 or all(bound is None for bound in bounds):
                raise ValueError(f"{name}_range must contain at least one finite bound")
            lower, upper = bounds
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(f"{name}_range must be an ordered (lower, upper) pair")
            if lower is not None:
                mask &= values >= lower
            if upper is not None:
                mask &= values <= upper
        if not np.any(mask):
            raise ValueError("No stars remain after the mass/feh/radius filters")
        fields = (
            "frame_id", "vsini", "vsini_err", "age", "age_err", "mass",
            "mass_err", "feh", "feh_err", "radius", "radius_err",
            "logage", "logage_err", "teff_mean", "teff_err", "feh_mean",
            "gmag3_mean", "gmag3_err", "parallax_mean", "parallax_err",
        )
        return type(self)(
            **{
                name: value[mask] if value is not None else None
                for name in fields
                for value in (getattr(self, name),)
            }
        )

    @classmethod
    def from_analysis_results(
        cls,
        frame_ids,
        root=ANALYSIS_ROOT,
        spec_label=SPEC_LABEL,
        iso_label=ISO_LABEL,
        progress=True,
        cache_dir=RESULT_ROOT / "cache",
        use_cache=True,
        refresh_cache=False,
    ):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Analysis root is not mounted or does not exist: {root}")
        normalized = [normalize_frame_id(x) for x in frame_ids]
        if not normalized:
            raise ValueError("frame_ids must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("frame_ids contains duplicates after normalization")

        cache_key = "\n".join(
            ["summary_with_iso_observables_v4", str(root.resolve()), spec_label, iso_label, *normalized]
        ).encode()
        cache_path = Path(cache_dir) / f"analysis_{hashlib.sha256(cache_key).hexdigest()[:16]}.npz"
        if use_cache and cache_path.is_file() and not refresh_cache:
            with np.load(cache_path, allow_pickle=False) as cached:
                return cls(**{name: cached[name] for name in cached.files})

        rows = []
        frame_iterator = tqdm(
            normalized,
            desc="Loading analysis results",
            unit="frame",
            disable=not progress,
        )
        for frame_id in frame_iterator:
            frame_dir = find_frame_dir(root, frame_id)
            spec_nc = result_path(frame_dir, spec_label)
            iso_nc = result_path(frame_dir, iso_label)
            vsini, vsini_err = posterior_mean_std(spec_nc, "vsini")
            teff_mean, teff_err = posterior_mean_std(spec_nc, "teff")
            feh, feh_err = posterior_mean_std(spec_nc, "feh")
            age, age_err = posterior_mean_std(iso_nc, "age")
            mass, mass_err = posterior_mean_std(iso_nc, "mass")
            radius, radius_err = posterior_mean_std(iso_nc, "radius")
            logage, logage_err = posterior_median_std(iso_nc, "logage")
            observations = isochrone_observations_from_metadata(
                frame_dir / iso_label / "job_metadata.json"
            )
            rows.append(
                (
                    frame_id, vsini, vsini_err, age, age_err, mass, mass_err,
                    feh, feh_err, radius, radius_err, logage, logage_err,
                    teff_mean, teff_err, feh,
                    observations["gmag3_mean"], observations["gmag3_err"],
                    observations["parallax_mean"], observations["parallax_err"],
                )
            )

        columns = list(zip(*rows))
        result = cls(columns[0], *columns[1:])
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                **{name: getattr(result, name) for name in result.__dataclass_fields__},
            )
        return result


@dataclass(frozen=True)
class IsochronePosterior:
    """Paired age/radius samples from each star's isochrone posterior."""

    frame_id: np.ndarray
    age: np.ndarray
    radius: np.ndarray

    def __post_init__(self):
        frame_id = np.asarray(self.frame_id, dtype=str)
        age = np.asarray(self.age, dtype=float)
        radius = np.asarray(self.radius, dtype=float)
        if frame_id.ndim != 1 or frame_id.size == 0:
            raise ValueError("frame_id must be a non-empty one-dimensional array")
        if age.ndim != 2 or radius.shape != age.shape or age.shape[0] != frame_id.size:
            raise ValueError("age and radius must have shape (N_stars, N_samples)")
        if not np.all(np.isfinite(age)) or not np.all(np.isfinite(radius)):
            raise ValueError("Isochrone posterior samples must be finite")
        if np.any(age <= 0) or np.any(radius <= 0):
            raise ValueError("Isochrone age and radius samples must be positive")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "age", age)
        object.__setattr__(self, "radius", radius)

    @property
    def N(self):
        return self.frame_id.size

    @property
    def Nsamples(self):
        return self.age.shape[1]

    @classmethod
    def from_analysis_results(
        cls,
        frame_ids,
        nsamples=10000,
        root=ANALYSIS_ROOT,
        iso_label=ISO_LABEL,
        progress=True,
        cache_dir=RESULT_ROOT / "cache",
        use_cache=True,
        refresh_cache=False,
    ):
        """Load paired age/radius draws, retaining their posterior ordering."""
        normalized = [normalize_frame_id(value) for value in frame_ids]
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("frame_ids must be non-empty and unique")
        if not isinstance(nsamples, int) or nsamples <= 0:
            raise ValueError("nsamples must be a positive integer")
        root = Path(root)
        cache_key = "\n".join(
            ["isochrone_posterior", str(root.resolve()), iso_label, str(nsamples), *normalized]
        ).encode()
        cache_path = Path(cache_dir) / f"iso_{hashlib.sha256(cache_key).hexdigest()[:16]}.npz"
        if use_cache and cache_path.is_file() and not refresh_cache:
            with np.load(cache_path, allow_pickle=False) as cached:
                return cls(**{name: cached[name] for name in cached.files})

        ages = []
        radii = []
        iterator = tqdm(
            normalized,
            desc="Loading isochrone posterior",
            unit="frame",
            disable=not progress,
        )
        for frame_id in iterator:
            iso_nc = result_path(find_frame_dir(root, frame_id), iso_label)
            posterior = az.from_netcdf(iso_nc).posterior
            missing = {"age", "radius"}.difference(posterior.data_vars)
            if missing:
                raise KeyError(f"Missing isochrone variables {sorted(missing)} in {iso_nc}")
            age = np.asarray(posterior["age"], dtype=float).reshape(-1)
            radius = np.asarray(posterior["radius"], dtype=float).reshape(-1)
            if age.size < nsamples:
                raise ValueError(f"{iso_nc} has {age.size} draws; requested {nsamples}")
            ages.append(age[:nsamples])
            radii.append(radius[:nsamples])

        result = cls(normalized, np.stack(ages), np.stack(radii))
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                frame_id=result.frame_id,
                age=result.age,
                radius=result.radius,
            )
        return result


def load_isochrone_posterior_means(
    frame_ids,
    *,
    root=ANALYSIS_ROOT,
    iso_label=ISO_LABEL,
    progress=True,
):
    """Return posterior-mean mass, age, [Fe/H], and radius per star."""
    normalized = [normalize_frame_id(frame_id) for frame_id in frame_ids]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("frame_ids must be non-empty and unique")

    rows = []
    iterator = tqdm(
        normalized,
        desc="Loading isochrone means",
        unit="star",
        disable=not progress,
    )
    for frame_id in iterator:
        iso_nc = result_path(find_frame_dir(Path(root), frame_id), iso_label)
        posterior = az.from_netcdf(iso_nc).posterior
        required = {"mass", "age", "feh", "radius"}
        missing = required.difference(posterior.data_vars)
        if missing:
            raise KeyError(f"Missing isochrone variables {sorted(missing)} in {iso_nc}")
        means = {
            name: float(np.asarray(posterior[name], dtype=float).mean())
            for name in ("mass", "age", "feh", "radius")
        }
        if not np.all(np.isfinite(list(means.values()))):
            raise ValueError(f"Non-finite isochrone posterior mean in {iso_nc}")
        rows.append({"frame_id": frame_id, **means})

    return pd.DataFrame(
        rows, columns=["frame_id", "mass", "age", "feh", "radius"]
    )
