import swiftsimio as sw
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hydra import initialize, compose
import os
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.data_fetcher import DataFetcherFactory


class TestFlamingoData(unittest.TestCase):
    def setUp(self):
        particles_file = '../../../../local/data/datasets/flamingo/L1000N0900/SOAP-HBT/flamingo_0073.hdf5'
        fof_subhalos_file = '../../../../local/data/datasets/flamingo/L1000N0900/SOAP-HBT/halo_properties_0073.hdf5'
        self.h = 0.6774

        #self.fof_subhalos = sw.load(fof_subhalos_file)
        #self.data = sw.load(particles_file)
        self.filtered_halos = None

    def test_load_fof(self):
        # get fof halo positions
        m200 = self.fof_subhalos.spherical_overdensity_200_crit.total_mass
        m200.convert_to_units('Msun')
        # Specify physical/comoving
        m200.convert_to_physical()
        m200 *= self.h

        r200 = self.fof_subhalos.spherical_overdensity_200_crit.soradius
        r200.convert_to_units('kpc')
        r200.convert_to_physical()
        r200 *= self.h

        halo_center = self.fof_subhalos.input_halos.halo_centre
        halo_center.convert_to_units('kpc')
        halo_center.convert_to_physical()
        halo_center *= self.h

        halo_id = np.arange(len(halo_center))

        # Count how many halos are after mass filtering
        are_massive = m200 > 1e13 * 0.6774
        are_primary = self.fof_subhalos.soap.host_halo_index == -1
        final_sample_size = are_massive & are_primary
        n = np.sum(final_sample_size)
        print(n)

        # Filtered values
        m200 = m200[final_sample_size]
        r200 = r200[final_sample_size]
        halo_center = halo_center[final_sample_size]
        halo_id = halo_id[final_sample_size]

        self.filtered_halos = pd.DataFrame(data=[], columns=[])
        self.filtered_halos['m200'] = m200
        self.filtered_halos['r200'] = r200
        self.filtered_halos['gpos'] = [halo_center[i] for i in range(halo_center.shape[0])]
        self.filtered_halos['halo_id'] = halo_id
        self.filtered_halos.to_csv('expected_halos_flamingo_L1000N0900.csv', index=False)
        print()

    def test_plot_fof_mass_num_subhalos(self):
        m200_array = self.fof_subhalos.spherical_overdensity_200_crit.total_mass
        m200_array.convert_to_units('Msun')
        m200_array.convert_to_physical()
        # m200_array *= self.h

        # Get subhalo primary ids
        mask = self.fof_subhalos.input_halos.is_central == 1
        subhalos_primary_idx_array = np.where(mask)[0]

        # test
        subhalos_primary_idx_array = subhalos_primary_idx_array[:5000]

        d = {}
        for subhalos_primary_idx in subhalos_primary_idx_array:
            if subhalos_primary_idx != 1:
                satellites_mask = self.fof_subhalos.soap.host_halo_index == subhalos_primary_idx
                satellites_idx = np.where(satellites_mask)[0]
                print(f'CORRECT - primary subhalo: {subhalos_primary_idx}, satellites: {satellites_idx}, fof_id')

                d[subhalos_primary_idx] = {}
                d[subhalos_primary_idx]['num_subhalos'] = len(satellites_idx) + 1
                d[subhalos_primary_idx]['m200'] = m200_array[subhalos_primary_idx].value

        df = pd.DataFrame.from_dict(d, orient="index")
        df.plot(x='m200', y='num_subhalos', kind='scatter', loglog=True)
        # Set labels and title
        plt.xlabel('Log(m200)')
        plt.ylabel('Log(num_subhalos)')
        plt.title('Logarithmic Scale Plot')

        # Save the plot to a file (e.g., PNG, PDF, etc.)
        plt.savefig('subhalos_vs_mass.png', dpi=300)  # Saves as PNG with high resolution
        plt.show()

    def test_get_subhalo_by_fof(self):
        # Get stellar mass, half_mass_radius_stars and coordinates for all the subhalos
        stellar_mass_array = self.fof_subhalos.bound_subhalo.stellar_mass
        stellar_mass_array.convert_to_units('Msun')
        stellar_mass_array.convert_to_physical()

        shmr_array = self.fof_subhalos.bound_subhalo.half_mass_radius_stars
        shmr_array.convert_to_units('kpc')
        shmr_array.convert_to_physical()
        shmr_array *= self.h

        coordinates_array = self.fof_subhalos.input_halos.halo_centre
        coordinates_array.convert_to_units('kpc')
        coordinates_array.convert_to_physical()
        coordinates_array *= self.h

        subhalo_id = np.arange(len(stellar_mass_array))
        id_parents = self.fof_subhalos.soap.host_halo_index.value

        # Filtered values
        are_massive = stellar_mass_array > 1e10
        are_child = self.fof_subhalos.soap.host_halo_index != -1
        final_sample_size = are_massive & are_child
        n = np.sum(final_sample_size)
        print(n)

        stellar_mass_array = stellar_mass_array[final_sample_size]
        shmr_array = shmr_array[final_sample_size]
        coordinates_array = coordinates_array[final_sample_size]
        subhalo_id = subhalo_id[final_sample_size]
        id_parents = id_parents[final_sample_size]

        filtered_subhalos = pd.DataFrame(data=[], columns=[])
        filtered_subhalos['mstr'] = stellar_mass_array
        filtered_subhalos['shmr'] = shmr_array
        filtered_subhalos['spos'] = [coordinates_array[i] for i in range(coordinates_array.shape[0])]
        filtered_subhalos['subhalo_id'] = subhalo_id
        filtered_subhalos['id_parents'] = id_parents
        filtered_subhalos.to_csv('expected_subhalos_flamingo_L1000N0900.csv', index=False)

        # --------
        fof_id = 4
        children_mask = id_parents == 4
        children_list = subhalo_id[children_mask]
        print()

    def test_load_particles(self):
        # Load data for star particles
        velocities_array = self.data.stars.velocities
        velocities_array.convert_to_units('km/s')
        velocities_array.convert_to_physical()

        coordinates_array = self.data.stars.coordinates
        coordinates_array.convert_to_units('kpc')
        coordinates_array.convert_to_physical()
        coordinates_array *= self.h

        mass_array = self.data.stars.masses
        mass_array.convert_to_units('Msun')
        mass_array.convert_to_physical()
        mass_array *= self.h

        # Star particles bound to the primary subhalo
        subhalos_primary_idx = 10
        fof_ids = sorted(list(set(self.fof_subhalos.input_halos.halo_catalogue_index.value)))
        unique_ids = sorted(list(set(self.data.stars.halo_catalogue_index.value)))
        particles_mask_primary = self.data.stars.halo_catalogue_index == subhalos_primary_idx

        # Get satellite subhalos
        satellites_mask = self.fof_subhalos.soap.host_halo_index == subhalos_primary_idx

        # Loop over the halo_catalogue_index of the satellites to get all bound particles
        satellites_idx_array = self.fof_subhalos.input_halos.halo_catalogue_index[satellites_mask]
        particles_mask_satellites = np.isin(self.data.stars.halo_catalogue_index, satellites_idx_array)

        particles_mask = particles_mask_primary | particles_mask_satellites

        velocity_particles = velocities_array[particles_mask]
        coordinates_particles = coordinates_array[particles_mask]
        mass_particles = mass_array[particles_mask]
        print(len(vel))

    def test_plot_fof_mass_particles(self):
        m200_array = self.fof_subhalos.spherical_overdensity_200_crit.total_mass
        m200_array.convert_to_units('Msun')
        # Specify physical/comoving
        m200_array.convert_to_physical()
        m200_array *= self.h

        mass_array = self.data.stars.masses
        mass_array.convert_to_units('Msun')
        mass_array.convert_to_physical()
        mass_array *= self.h

        # Get subhalo primary ids
        mask = self.fof_subhalos.input_halos.is_central == 1
        subhalos_primary_idx_array = np.where(mask)[0]

        # test
        subhalos_primary_idx_array = subhalos_primary_idx_array[:100]

        d = {}
        for subhalos_primary_idx in subhalos_primary_idx_array:
            if subhalos_primary_idx != 1:
                d[fof_id] = {}

                # Star particles bound to the primary subhalo
                particles_mask_primary = self.data.stars.halo_catalogue_index == subhalos_primary_idx

                # Get satellite subhalos
                satellites_mask = self.fof_subhalos.soap.host_halo_index == subhalos_primary_idx

                # Loop over the halo_catalogue_index of the satellites to get all bound particles
                satellites_idx_array = self.fof_subhalos.input_halos.halo_catalogue_index[satellites_mask]
                particles_mask_satellites = np.isin(self.data.stars.halo_catalogue_index, satellites_idx_array)

                particles_mask = particles_mask_primary | particles_mask_satellites
                mass_particles_fof = mass_array[particles_mask]

                d[fof_id]['particles_mass'] = np.sum(mass_particles_fof)
                d[fof_id]['m200'] = m200_array[fof_id].value

        df = pd.DataFrame.from_dict(d, orient="index")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["m200"], df["particles_mass"], color="b", alpha=0.7, label="Particle mass")
        ax.set_xlabel("M200")
        ax.set_ylabel("Particle mass")
        ax.set_title("Particles mass as a Function of FoF Mass")
        ax.legend()
        ax.grid(True)
        fig.savefig("particles_vs_mass.png", dpi=300, bbox_inches="tight")  # Save as PNG
        plt.show()

    def test_fetcher_flamingo(self):
        with initialize(config_path="../../../../configs"):
            self.cfg_flamingo = compose(config_name="data_generation.yaml")
            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_flamingo.paths = local_config.paths
            self.cfg_flamingo.catalog = compose("catalog/flamingo.yaml")['catalog']
            self.cfg_flamingo.catalog.path = '../../../../local/data/datasets/flamingo/L1000N0900'

        os.environ['PROJECT_ROOT'] = '../../..'

        required_keys = ['G']
        data_loader_dict = next((step for step in self.cfg_flamingo.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        self.fetcher_flamingo = DataFetcherFactory.get_data_fetcher(self.cfg_flamingo.catalog, **attr_dict)

        # Non filtered primary halos ids
        expected_num_halos = 128967
        all_primary_halos_id_mask = self.fof_subhalos.soap.host_halo_index == -1

        # Halos
        halo_context_1 = self.fetcher_flamingo.get_halos()
        halos_df = halo_context_1.data
        halos_df = halos_df.reset_index(drop=True)
        halos_df = halos_df.sort_values(by='halo_id')
        halos_df.to_csv('halos_flamingo_L1000N0900.csv', index=False)

        #halos_df = pd.read_csv('halos_flamingo_L1000N0900.csv')
        id_primary_halos = halos_df['halo_id'].unique()
        self.assertEqual(len(id_primary_halos), expected_num_halos)

        # Subhalos
        expected_num_subhalos = 2084586
        subhalo_context_1 = self.fetcher_flamingo.get_subhalos()
        subhalos_df = subhalo_context_1.data
        subhalos_df = subhalos_df.reset_index(drop=True)
        subhalos_df = subhalos_df.sort_values(by='id_parents')
        subhalos_df.to_csv('subhalos_flamingo_L1000N0900.csv', index=False)
        #subhalos_df = pd.read_csv('subhalos_flamingo_L1000N0900.csv')
        id_parents = subhalos_df['id_parents']

        error_parents = set(id_parents) - set(id_primary_halos)

        # Example case fof=4
        expected_child_list = [4, 20, 75, 112, 162]
        child_list = subhalos_df[subhalos_df['id_parents'] == 4]['subhalo_id'].tolist()
        tmp = subhalo_context_1.get_items_by_halo(4)
        self.assertSetEqual(set(expected_child_list), set(child_list))

    def test_particle_distribution(self):
        df = pd.read_csv('../../../../transition_radius_exploration/flamingo_L1000N0900/_flamingo_L1000N0900_transition_radius.csv')
        tmp = df.sort_values(by='number_particles', ascending=False).head(n=5)
        print(list(df.sort_values(by='number_particles', ascending=False).head(n=5)['halo_id']))






