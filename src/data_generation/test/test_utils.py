import unittest

import joblib
import numpy as np
import pandas as pd
from hydra import initialize, compose
from matplotlib import pyplot as plt
from rootutils import rootutils
import seaborn as sns

rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.utils import read_dataset, to_list, to_list_in_batches, plot_transition_radius, \
    store_overlapping_matrix_as_df


def assert_dicts_equal(dict1, dict2, atol=1e-8):
    # Ensure the dictionaries have the same keys
    assert dict1.keys() == dict2.keys(), "Dictionaries have different keys"

    # Iterate over the keys and compare the arrays
    for key in dict1:
        arr1 = dict1[key]
        arr2 = dict2[key]

        # Check if the arrays are equal (use np.allclose for floating-point arrays)
        assert np.allclose(arr1, arr2, atol=atol), f"Arrays for key '{key}' are not equal"


class TestUtils(unittest.TestCase):
    def test_get_particle_data(self):
        # https://arxiv.org/pdf/1706.09899
        itype = 4   # PartType
        path = '../../../local/data/datasets/eagle/RefL0050N0752'
        gn = 1
        sgn = 0
        gns = read_dataset(itype, 'GroupNumber', path)
        sgns = read_dataset(itype, 'SubGroupNumber', path)
        mask = np.logical_and(gns == gn, sgns == sgn)

        data = {
            'mass': read_dataset(itype, 'Mass', path)[mask],
            'coords': read_dataset(itype, 'Coordinates', path)[mask],
            'velocity': read_dataset(itype, 'Velocity', path)[mask]
        }
        print(len(data))

    def test_to_list_batches(self):
        random_array = np.random.rand(10, 3)
        col = 'random_column'
        d = {col: random_array}
        converted = to_list(d, col)
        converted_ii = to_list_in_batches(d, col, batch_size=2)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(converted, converted_ii)))

    def test_find_values(self):
        comoving_resolution = 22.3 # for flamingo
        scale_factor = 0.8333333333333335
        spatial_resolution = comoving_resolution / scale_factor # Convert to physical kpc
        print(spatial_resolution)

        # bin calculation
        fov = 429352.020 * 2
        r = fov / 2  # Radius of the field of view

        bin_size = spatial_resolution * 3
        bins = int(fov / bin_size)
        print(f'spatial resolution: {spatial_resolution}, fov: {fov}, r: {r}, bins: {bins}, bin size: {bin_size}')

    def test_ckdtree(self):
        import numpy as np
        from scipy.spatial import cKDTree

        positions = np.array([
            [0, 0, 0],  # index 0 → maybe halo ID 100
            [0, 1, 0],  # index 1 → maybe halo ID 101
            [2, 2, 2],  # index 2 → maybe halo ID 102
        ])
        halo_ids = np.array([100, 101, 102])

        tree = cKDTree(positions)

        # Query neighbors within distance 1.5 of point [0,0,0]
        indices = tree.query_ball_point([0, 0, 0], r=1.5)

        print(indices)  # ➝ [0, 1]
        print(halo_ids[indices])  # ➝ [100, 101] ← you get the actual halo IDs here

    def test_plot_transition_radius(self):
        # Initialize Hydra and load configuration
        with initialize(config_path="configs"):
            # Initialize the cfg_illustris and cfg_eagle
            cfg_illustris = compose(config_name="data_generation.yaml")
            cfg_eagle = compose(config_name="data_generation.yaml")
            cfg_flamingo = compose(config_name="data_generation.yaml")
            cfg_millenium = compose(config_name="data_generation.yaml")

        transition_radius_list = np.arange(10)
        plot_transition_radius(cfg_illustris, transition_radius_list, log_normalization=True)

    def test_overlapping_matrix_as_df(self):
        overlapping_matrix = {
            1: [(3, 101), (4, 102)],
            2: [(5, 201)]
        }
        # Initialize Hydra and load configuration
        with initialize(config_path="configs"):
            # Initialize the cfg_illustris and cfg_eagle
            cfg_illustris = compose(config_name="data_generation.yaml")
            cfg_eagle = compose(config_name="data_generation.yaml")
            cfg_flamingo = compose(config_name="data_generation.yaml")
            cfg_millenium = compose(config_name="data_generation.yaml")

        store_overlapping_matrix_as_df(cfg_illustris, overlapping_matrix)

    def test_filter_outliers(self):
        df = pd.read_csv('../../../transition_radius_exploration/flamingo_L1000N0900/_flamingo_L1000N0900_transition_radius.csv')
        transition_radius = df['transition_radius']
        tmp = df.describe()
        percentiles = [97]
        ignored_halos = []

        for percentile in percentiles:
            upper_bound = np.percentile(transition_radius, percentile)
            lower_bound = 100 - upper_bound
            filtered_df = df[(transition_radius >= lower_bound) & (transition_radius <= upper_bound)]
            print(f'Percentile: {percentile}, transition radius [kpc]: {upper_bound}, final size: {len(filtered_df)}, Ignored halos: {len(df) - len(filtered_df)}')
            ignored_halos = set(df['halo_id']) - set(filtered_df['halo_id'])
            print(ignored_halos)

            sns.histplot(filtered_df["transition_radius"]/filtered_df['R200'], bins=100, kde=True)
            plt.title(f"Transition radius/R200 distribution, percentile: {percentile}")
            #plt.show()

        ignored_halos = list(ignored_halos)
        new_df = df[~df['halo_id'].isin(ignored_halos)]
        sts = new_df.describe()
        print()

        errors_df = pd.read_csv('../../../transition_radius_exploration/flamingo_L1000N0900/errors__flamingo_L1000N0900.csv')
        halos = errors_df['halo_id']
        final_set = set(ignored_halos).union(set(halos))
        print()

    def test_explore_problematic_halos(self):
        # What does it have in common halos that don't have TR ?
        df = pd.read_csv('../../../transition_radius_exploration/illustris/errors__illustris300.csv')
        halos_id = df['halo_id']
        raw_halos = pd.read_csv('../../../illustris_halo_context_raw.csv')
        problematic_halos = raw_halos[raw_halos['halo_id'].isin(halos_id)]
        good_halos = raw_halos[~raw_halos['halo_id'].isin(halos_id)]

        np.random.seed(0)

        features = ['m200']

        df1_long = problematic_halos[features].assign(dataset="problematic halos")
        df2_long = good_halos[features].assign(dataset="good halos")
        long_df = pd.concat([df1_long, df2_long], ignore_index=True)
        long_df = long_df.melt(id_vars="dataset", var_name="feature", value_name="value")
        plt.figure(figsize=(10, 6))

        datasets = long_df["dataset"].unique()
        features_unique = long_df["feature"].unique()

        positions = []
        data = []
        pos = 1
        width = 0.35

        for feature in features_unique:
            for i, ds in enumerate(datasets):
                values = long_df[
                    (long_df["feature"] == feature) &
                    (long_df["dataset"] == ds)
                    ]["value"].dropna()

                data.append(values)
                positions.append(pos + i * width)
            pos += 1.5  # space between features

        plt.boxplot(data, positions=positions, widths=0.3)

        tick_positions = []
        pos = 1
        for _ in features_unique:
            tick_positions.append(pos + width / 2)  # middle of each feature's box pair
            pos += 1.5

        plt.xticks(tick_positions, features_unique, rotation=30)

        plt.ylabel("Value")
        plt.title("Feature comparison between problematic halos and good halos")

        # Add legend manually
        for i, ds in enumerate(datasets):
            plt.plot([], c='k', label=ds)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Data
        x1 = problematic_halos['r200'].dropna()
        x2 = good_halos['r200'].dropna()

        # -----------------------
        # Figure 1: Problematic halos
        # -----------------------
        plt.figure(figsize=(8, 5))

        # Logarithmic bins (optional but better for mass distributions)
        bins1 = np.logspace(np.log10(x1.min()), np.log10(x1.max()), 20)

        plt.hist(x1, bins=bins1, alpha=0.7, color="blue", edgecolor="black")
        plt.xscale("log")
        plt.xlabel("r200")
        plt.ylabel("Count")
        plt.title("Distribution of r200 for Problematic Halos")
        plt.tight_layout()
        plt.savefig("problematic_halos_r200.png", dpi=300)  # dpi=300 for high quality
        plt.show()

        # -----------------------
        # Figure 2: Good halos
        # -----------------------
        plt.figure(figsize=(8, 5))

        bins2 = np.logspace(np.log10(x2.min()), np.log10(x2.max()), 20)

        plt.hist(x2, bins=bins2, alpha=0.7, color="red", edgecolor="black")
        plt.xscale("log")
        plt.xlabel("r200")
        plt.ylabel("Count")
        plt.title("Distribution of r200 for Good Halos")
        plt.tight_layout()
        plt.savefig("good_halos_r200.png", dpi=300)  # dpi=300 for high quality
        plt.show()

    def test_eagle_RefL0100N1504(self):
        halo_subhalo_list = joblib.load('../../../transition_radius_exploration/halo_subhalo_list_RefL0100N1504.joblib')
        print()









