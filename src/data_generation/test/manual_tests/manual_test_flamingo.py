import swiftsimio as sw
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TestFlamingoData(unittest.TestCase):
    def setUp(self):
        particles_file = '../../../../local/data/datasets/flamingo/L1000N0900/SOAP-HBT/flamingo_0073.hdf5'
        fof_subhalos_file = '../../../../local/data/datasets/flamingo/L1000N0900/SOAP-HBT/halo_properties_0073.hdf5'
        self.h = 0.6774

        self.fof_subhalos = sw.load(fof_subhalos_file)
        self.data = sw.load(particles_file)

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

        halo_center = self.fof_subhalos.input_halos_fof.centres
        halo_center.convert_to_units('kpc')
        halo_center.convert_to_physical()
        halo_center *= self.h

        n = len(self.fof_subhalos.spherical_overdensity_200_crit.soradius)
        primary_subhalos_id = np.arange(n)

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
        fof_id = 4

        # Get stellar mass, half_mass_radius_stars and coordinates for all the subhalos
        stellar_mass_array = self.fof_subhalos.bound_subhalo.stellar_mass
        stellar_mass_array.convert_to_units('Msun')
        stellar_mass_array.convert_to_physical()
        stellar_mass_array *= self.h

        shmr_array = self.fof_subhalos.bound_subhalo.half_mass_radius_stars
        shmr_array.convert_to_units('kpc')
        shmr_array.convert_to_physical()
        shmr_array *= self.h

        coordinates_array = self.fof_subhalos.input_halos.halo_centre
        coordinates_array.convert_to_units('kpc')
        coordinates_array.convert_to_physical()
        coordinates_array *= self.h

        # Retrieve data for the primary subhalo, correct
        stellar_mass = stellar_mass_array[fof_id]
        shmr = shmr_array[fof_id]
        coordinates = coordinates_array[fof_id]

        # TODO Retrieve data for the subhalo satellites, correct
        satellites_mask = self.fof_subhalos.soap.host_halo_index == fof_id
        n_satellites = np.sum(satellites_mask)

        satellite_coordiantes = coordinates_array[satellites_mask]
        satellite_shmr = shmr_array[satellites_mask]
        satellite_stellar_mass = stellar_mass_array[satellites_mask]
        print(stellar_mass, shmr, coordinates)

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
        subhalos_primary_idx = 4
        particles_mask_primary = self.data.stars.halo_catalogue_index == subhalos_primary_idx

        # Get satellite subhalos
        satellites_mask = self.fof_subhalos.soap.host_halo_index == subhalos_primary_idx

        # Loop over the halo_catalogue_index of the satellites to get all bound particles
        satellites_idx_array = self.fof_subhalos.input_halos.halo_catalogue_index[satellites_mask]
        particles_mask_satellites = self.data.stars.halo_catalogue_index == satellites_idx_array

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
                particles_mask_satellites = self.data.stars.halo_catalogue_index == satellites_idx_array

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


