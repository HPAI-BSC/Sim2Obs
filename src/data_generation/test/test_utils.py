import unittest

from src.data_generation.src.utils import compute_intrahalo_variance


class TestUtils(unittest.TestCase):
    def test_compute_intrahalo_variance(self):
        compute_intrahalo_variance(input_dir='.')