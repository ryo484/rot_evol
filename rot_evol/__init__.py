"""Public compatibility facade for rotation-evolution analysis.

Implementation is split across ``rot_evol.data``, ``rot_evol.models``, and
``rot_evol.plotting``. Existing imports from ``rot_evol`` remain supported.
"""

from .data import (
    ANALYSIS_ROOT,
    ISO_LABEL,
    RESULT_ROOT,
    SPEC_LABEL,
    IsochronePosterior,
    RotEvol,
    find_frame_dir,
    frame_ids_from_observation_log,
    isochrone_observations_from_metadata,
    load_isochrone_posterior_means,
    normalize_frame_id,
    posterior_mean_std,
    posterior_median_std,
    result_path,
)
from .models import (
    DAY_S,
    R_SUN_KM,
    broken_power_law,
    evaluate_rotation_law,
    gp_rotation_law,
    jaxspin_rotation_law,
    loglike_vsini_single,
    normalize_rotation_law,
    pi0_linear,
    rotation_parameter_names,
    rotation_slope_names,
    v_power,
)
from .plotting import (
    infer_rotation_law,
    posterior_rotation_draws,
    save_figures,
    save_inference_data,
    save_isochrone_mean_triangle,
    save_logage_histogram,
    save_logage_posterior_histogram,
    selection_label,
)

__all__ = [
    "ANALYSIS_ROOT", "SPEC_LABEL", "ISO_LABEL", "RESULT_ROOT",
    "R_SUN_KM", "DAY_S", "RotEvol", "IsochronePosterior",
    "normalize_frame_id", "posterior_mean_std", "posterior_median_std",
    "find_frame_dir", "result_path", "frame_ids_from_observation_log",
    "isochrone_observations_from_metadata",
    "load_isochrone_posterior_means", "save_isochrone_mean_triangle",
    "selection_label", "save_logage_histogram",
    "save_logage_posterior_histogram", "v_power",
    "broken_power_law", "gp_rotation_law", "jaxspin_rotation_law",
    "normalize_rotation_law", "rotation_parameter_names",
    "rotation_slope_names",
    "evaluate_rotation_law", "pi0_linear", "loglike_vsini_single",
    "save_inference_data", "infer_rotation_law",
    "posterior_rotation_draws", "save_figures",
]
