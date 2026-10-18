import unittest

import pandas as pd

from mta import MTA, MTAConfig


class OptionalExposureTimesTest(unittest.TestCase):
    def test_missing_exposure_times_is_allowed_when_generation_is_disabled(self):
        data = pd.DataFrame(
            {
                "path": ["a > a > b", "a > b"],
                "total_conversions": [1, 2],
                "total_null": [3, 4],
            }
        )

        mta = MTA(data, config=MTAConfig(add_timepoints=False, allow_loops=False))

        self.assertNotIn("exposure_times", mta.data.columns)
        self.assertEqual(mta.data["path"].tolist(), [["a", "b"]])
        self.assertEqual(mta.data["total_conversions"].tolist(), [3])
        self.assertEqual(mta.data["total_null"].tolist(), [7])

        mta.first_touch().last_touch().linear()

    def test_time_dependent_model_explains_missing_exposure_times(self):
        data = pd.DataFrame(
            {
                "path": ["a > b"],
                "total_conversions": [1],
                "total_null": [0],
            }
        )
        mta = MTA(data, config=MTAConfig(add_timepoints=False))

        with self.assertRaisesRegex(ValueError, "additive_hazard requires exposure_times"):
            mta.additive_hazard()

    def test_mismatched_exposure_times_length_is_rejected(self):
        """A path with 3 nodes but 1 timestamp must not raise IndexError.

        Regression test: remove_loops indexed exposure_times by path position,
        so a short exposure_times column crashed construction with an opaque
        IndexError instead of an actionable message.
        """
        data = pd.DataFrame(
            {
                "path": ["a > b > c"],
                "total_conversions": [1],
                "total_null": [0],
                "exposure_times": ["2020-01-01 00:00:00"],
            }
        )

        with self.assertRaisesRegex(ValueError, "exposure_times has 1 timestamp"):
            MTA(data)

    def test_excess_exposure_times_length_is_rejected(self):
        data = pd.DataFrame(
            {
                "path": ["a > b"],
                "total_conversions": [1],
                "total_null": [0],
                "exposure_times": [
                    "2020-01-01 00:00:00 > 2020-01-01 00:00:05 > 2020-01-01 00:00:09"
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "path has 2 element"):
            MTA(data)

    def test_aligned_exposure_times_are_accepted(self):
        data = pd.DataFrame(
            {
                "path": ["a > b > c"],
                "total_conversions": [1],
                "total_null": [0],
                "exposure_times": [
                    "2020-01-01 00:00:00 > 2020-01-01 00:00:05 > 2020-01-01 00:00:09"
                ],
            }
        )

        mta = MTA(data)

        self.assertEqual(mta.data["path"].tolist(), [["a", "b", "c"]])
        self.assertEqual(len(mta.data["exposure_times"].tolist()[0]), 3)
        mta.additive_hazard(epochs=2)

    def test_empty_path_with_empty_exposure_times_is_accepted(self):
        """Blank path and blank exposure_times both split to zero elements."""
        data = pd.DataFrame(
            {
                "path": ["", "a > b"],
                "total_conversions": [1, 1],
                "total_null": [0, 0],
                "exposure_times": ["", "2020-01-01 00:00:00 > 2020-01-01 00:00:05"],
            }
        )

        mta = MTA(data)

        self.assertEqual(mta.data["path"].tolist(), [[], ["a", "b"]])


if __name__ == "__main__":
    unittest.main()
