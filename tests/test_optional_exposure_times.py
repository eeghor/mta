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


if __name__ == "__main__":
    unittest.main()
