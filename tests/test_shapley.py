import unittest

import pandas as pd

from mta import MTA


class ShapleyAttributionTest(unittest.TestCase):
    def test_single_channel_gets_all_credit(self):
        data = pd.DataFrame(
            {
                "path": ["a"],
                "total_conversions": [5],
                "total_null": [0],
            }
        )

        mta = MTA(data)
        mta.shapley(normalize=False)

        self.assertEqual(dict(mta.attribution["shapley"]), {"a": 5.0})

    def test_four_channel_path_preserves_total_conversions(self):
        data = pd.DataFrame(
            {
                "path": ["a > b > c > d"],
                "total_conversions": [10],
                "total_null": [0],
            }
        )

        mta = MTA(data)
        mta.shapley(normalize=False)

        shapley = dict(mta.attribution["shapley"])
        self.assertAlmostEqual(sum(shapley.values()), 10.0)
        self.assertEqual(shapley, {"a": 2.5, "b": 2.5, "c": 2.5, "d": 2.5})

    def test_mixed_paths_allocate_marginal_conversions(self):
        data = pd.DataFrame(
            {
                "path": ["a", "a > b"],
                "total_conversions": [4, 6],
                "total_null": [0, 0],
            }
        )

        mta = MTA(data)
        mta.shapley(normalize=False)

        self.assertEqual(dict(mta.attribution["shapley"]), {"a": 7.0, "b": 3.0})


if __name__ == "__main__":
    unittest.main()
