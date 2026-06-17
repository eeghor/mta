import unittest

import pandas as pd

from mta import MTA, MTAConfig


class CustomSeparatorTest(unittest.TestCase):
    def test_custom_separator_is_used_for_generated_exposure_times(self):
        data = pd.DataFrame(
            {
                "path": ["a | a | b"],
                "total_conversions": [1],
                "total_null": [0],
            }
        )

        mta = MTA(data, config=MTAConfig(sep=" | ", allow_loops=False))

        self.assertEqual(mta.data["path"].tolist(), [["a", "b"]])
        self.assertEqual(len(mta.data["exposure_times"].iloc[0]), 2)
        self.assertEqual(mta.channels, ["a", "b"])

    def test_custom_separator_is_used_when_removing_loops(self):
        data = pd.DataFrame(
            {
                "path": ["a | a | b"],
                "exposure_times": [
                    "2024-01-01 00:00:00 | "
                    "2024-01-01 00:00:01 | "
                    "2024-01-01 00:00:02"
                ],
                "total_conversions": [1],
                "total_null": [0],
            }
        )

        mta = MTA(data, config=MTAConfig(sep=" | ", allow_loops=False))

        self.assertEqual(mta.data["path"].tolist(), [["a", "b"]])
        self.assertEqual(
            mta.data["exposure_times"].tolist(),
            [["2024-01-01 00:00:00", "2024-01-01 00:00:02"]],
        )


if __name__ == "__main__":
    unittest.main()
