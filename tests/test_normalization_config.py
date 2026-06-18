import unittest

import pandas as pd

from mta import MTA, MTAConfig


class NormalizationConfigTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "path": ["a", "b"],
                "total_conversions": [2, 1],
                "total_null": [0, 0],
            }
        )

    def test_config_disables_normalization_by_default(self):
        mta = MTA(
            self.data,
            config=MTAConfig(normalize_by_default=False),
        )

        mta.first_touch().shapley()

        self.assertEqual(mta.attribution["first_touch"], {"a": 2, "b": 1})
        self.assertEqual(mta.attribution["shapley"], {"a": 2.0, "b": 1.0})

    def test_explicit_normalize_overrides_config(self):
        mta = MTA(
            self.data,
            config=MTAConfig(normalize_by_default=False),
        )

        mta.first_touch(normalize=True)

        self.assertEqual(
            mta.attribution["first_touch"],
            {"a": 0.666667, "b": 0.333333},
        )


if __name__ == "__main__":
    unittest.main()
