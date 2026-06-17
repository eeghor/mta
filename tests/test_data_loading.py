import os
import tempfile
import unittest

from mta import MTA


class DataLoadingTest(unittest.TestCase):
    def test_relative_csv_path_loads_from_current_working_directory(self):
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "customer_journeys.csv")
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("path,total_conversions,total_null\n")
                fh.write("local_a > local_b,3,1\n")

            try:
                os.chdir(tmpdir)
                mta = MTA("customer_journeys.csv")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(mta.channels, ["local_a", "local_b"])
        self.assertEqual(mta.data["total_conversions"].tolist(), [3])

    def test_bundled_data_still_loads_when_relative_file_is_absent(self):
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                mta = MTA("data.csv.gz")
            finally:
                os.chdir(original_cwd)

        self.assertGreater(len(mta.data), 0)


if __name__ == "__main__":
    unittest.main()
