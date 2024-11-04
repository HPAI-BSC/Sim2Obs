import unittest

from src.utils import find_duplicates_in_halo


class TestDuplicates(unittest.TestCase):
    def test_find_duplicates_in_halo(self):
        halo_directory = "../data/images/labels"
        duplicates = find_duplicates_in_halo(halo_directory, idhalo=0)

        if duplicates:
            print("Duplicated images found:")
            for original, duplicate in duplicates:
                print(f"Original: {original}, Duplicate: {duplicate}")
        else:
            print("No duplicates found.")