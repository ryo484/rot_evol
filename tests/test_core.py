import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from rot_evol import (
    RotEvol,
    broken_power_law,
    isochrone_observations_from_metadata,
    normalize_frame_id,
    normalize_rotation_law,
    selection_label,
    v_power,
)


class RotationLawTests(unittest.TestCase):
    def test_power_law_is_normalized_at_solar_age(self):
        self.assertAlmostEqual(float(v_power(4.6, -0.5, 2.0, 1.0)), 3.0)

    def test_broken_power_law_is_continuous_at_break(self):
        value = broken_power_law(np.array([4.6]), -0.5, 0.1, 2.5, 4.6)
        np.testing.assert_allclose(np.asarray(value), [2.5])

    def test_rotation_law_aliases(self):
        self.assertEqual(normalize_rotation_law("broken"), "broken_power")
        with self.assertRaises(ValueError):
            normalize_rotation_law("unknown")


class DataTests(unittest.TestCase):
    def make_data(self):
        values = np.array([1.0, 1.2])
        errors = np.array([0.1, 0.1])
        return RotEvol(
            frame_id=np.array(["00000001", "00000002"]),
            vsini=np.array([2.0, 3.0]),
            vsini_err=errors,
            age=np.array([4.0, 5.0]),
            age_err=errors,
            mass=values,
            mass_err=errors,
            feh=np.array([-0.2, 0.1]),
            feh_err=errors,
            radius=values,
            radius_err=errors,
        )

    def test_selection_preserves_matching_rows(self):
        selected = self.make_data().select(
            mass_range=(0.9, 1.1), feh_range=(None, -0.1)
        )
        self.assertEqual(selected.frame_id.tolist(), ["00000001"])

    def test_metadata_loader_uses_corrected_magnitude(self):
        payload = {
            "gaia_dr3": {
                "gmag3_extinction_corrected": 9.1,
                "gmag3_extinction_corrected_err": 0.02,
                "gaia_parallax_mas": 2.3,
                "gaia_parallax_error_mas": 0.04,
            }
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "job_metadata.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = isochrone_observations_from_metadata(path)
        self.assertEqual(result["gmag3_mean"], 9.1)
        self.assertEqual(result["parallax_mean"], 2.3)

    def test_labels_and_frame_normalization(self):
        self.assertEqual(normalize_frame_id("GRA123"), "00000123")
        self.assertEqual(
            selection_label((0.9, 1.1), (-0.1, None)),
            "mass0.9-1.1_feh-0.1up",
        )


if __name__ == "__main__":
    unittest.main()
