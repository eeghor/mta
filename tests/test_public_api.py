import unittest

from mta import MTA, MTAConfig


class PublicApiTest(unittest.TestCase):
    def test_readme_configuration_import(self):
        config = MTAConfig()

        self.assertEqual(MTA.__name__, "MTA")
        self.assertFalse(config.allow_loops)


if __name__ == "__main__":
    unittest.main()
