import os
import pickle
import unittest

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose
import eagleSqlTools as sql
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection

load_dotenv('../.env')


class TestFetcher(unittest.TestCase):
    def setUp(self):
        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        with open("data/illustris/subhalo_0.pkl", "rb") as file:
            self.subhalo_illustris = pickle.load(file)

        with open("data/illustris/halo_context_0.pkl", "rb") as file:
            self.halo_context_illustris = pickle.load(file)

        with open("data/illustris/subhalo_context_0.pkl", "rb") as file:
            self.subhalo_context_illustris = pickle.load(file)

        # Initialize Hydra and load configuration
        with initialize(config_path="configs"):
            # Initialize the cfg_illustris and cfg_eagle
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths
            self.cfg_eagle.paths = local_config.paths

            # Dynamic catalog loading for both Illustris and Eagle
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

        os.environ['PROJECT_ROOT'] = '../../..'

        # TODO update
        required_keys = ['G']
        data_loader_dict = next((step for step in self.cfg_illustris.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        self.fetcher_illustris = DataFetcherFactory.get_data_fetcher(self.cfg_illustris.catalog, **attr_dict)
        self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(self.cfg_eagle.catalog, **attr_dict)

    def test_get_halos_illustris(self):
        halo_context_1 = self.fetcher_illustris.get_halos()
        self.assertIsInstance(halo_context_1, HaloContext)
        self.assertEqual(len(halo_context_1.data), 83)

        # Reorder columns for the test
        tmp = halo_context_1.data.head()[['m200', 'r200', 'fsub', 'halo_id', 'gpos']]
        assert_frame_equal(tmp, self.halo_context_illustris.data, check_dtype=False)

        # get_item
        halo_data_1 = halo_context_1.get_halo_by_id(halo_id=0)
        self.assertEqual(halo_data_1.halo_id, self.halo_illustris.halo_id)
        self.assertEqual(halo_data_1.fsub, self.halo_illustris.fsub)
        self.assertAlmostEquals(halo_data_1.m200, self.halo_illustris.m200)
        self.assertAlmostEquals(halo_data_1.r200, self.halo_illustris.r200)
        assert_array_equal(halo_data_1.gpos, self.halo_illustris.gpos)

    def test_get_subhalos_illustris(self):
        subhalo_context_1 = self.fetcher_illustris.get_subhalos()
        self.assertIsInstance(subhalo_context_1, SubhaloContext)
        self.assertEqual(len(subhalo_context_1.data), 2598)

        tmp = subhalo_context_1.data.head()[['shmr', 'mstr', 'id_parents', 'subhalo_id', 'spos']]
        assert_frame_equal(tmp, self.subhalo_context_illustris.data, check_dtype=False)

        # Get item
        subhalo_data_1 = subhalo_context_1.get_subhalo_by_id(subhalo_id=0)
        self.assertEqual(subhalo_data_1.subhalo_id, self.subhalo_illustris.subhalo_id)
        self.assertEqual(subhalo_data_1.id_parents, self.subhalo_illustris.id_parents)
        self.assertAlmostEquals(subhalo_data_1.mstr, self.subhalo_illustris.mstr)
        self.assertAlmostEquals(subhalo_data_1.shmr, self.subhalo_illustris.shmr)
        assert_array_equal(subhalo_data_1.spos, self.subhalo_illustris.spos)

    def test_get_p_subhalo_illustris(self):
        subhalo_particles_1 = self.fetcher_illustris.get_p_subhalo(subhalo_id=10)

        self.assertIsInstance(subhalo_particles_1, SubhaloParticles)
        self.assertTrue(hasattr(subhalo_particles_1, "gas_m"))
        self.assertIsInstance(subhalo_particles_1.gas_m, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_1, "star_m"))
        self.assertIsInstance(subhalo_particles_1.star_m, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_1, "star_p"))
        self.assertIsInstance(subhalo_particles_1.star_p, np.ndarray)
        self.assertTrue(hasattr(subhalo_particles_1, "star_v"))
        self.assertIsInstance(subhalo_particles_1.star_v, np.ndarray)

    def test_get_halos_eagle(self):
        halo_context_2 = self.fetcher_eagle.get_halos()
        self.assertIsInstance(halo_context_2, HaloContext)
        # TODO
        #self.assertEqual(len(halo_context_2.data), XX)
        # get_item
        halo_data_2 = halo_context_2.get_item(halo_id=halo_context_2.halo_id[0])
        assert halo_data_2 is not None

    def test_get_subhalos_eagle(self):
        subhalo_context_2 = self.fetcher_eagle.get_subhalos()
        self.assertIsInstance(subhalo_context_2, SubhaloContext)

        # Get item
        subhalo_data_2 = subhalo_context_2.get_item(subhalo_id=subhalo_context_2.subhalo_id[0])
        assert subhalo_data_2 is not None


    def test_get_p_subhalo_eagle(self):
        self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
        self.cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'
        subhalo_particles_2 = self.fetcher_eagle.get_p_subhalo(subhalo_id=0, group_number=2)
        self.assertIsInstance(subhalo_particles_2, SubhaloParticles)
        self.assertTrue(hasattr(subhalo_particles_2, "gas_m"))
        self.assertTrue(hasattr(subhalo_particles_2, "star_m"))
        self.assertTrue(hasattr(subhalo_particles_2, "star_p"))
        self.assertTrue(hasattr(subhalo_particles_2, "star_v"))

    def test_eagle_simulation_codes(self):
        simulation_codes = ['RefL0025N0376',    # nok
                            'RefL0025N0752',    # nok
                            'RecalL0025N0752',  # nok
                            'RefL0050N0752',    # ok
                            'AGNdT9L0050N0752', # ok
                            'RefL0100N1504']    # ok

        for sim in simulation_codes:
            try:
                print(sim)
                self.cfg_eagle.catalog['simulation_code'] = sim
                self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(self.cfg_eagle)
                halos = self.fetcher_eagle.get_halos()
                print(f'-- Number of halos: {len(halos.data["m200"])}')
                subhalos = self.fetcher_eagle.get_subhalos()
                print(f'-- Number of subhalos: {len(subhalos.data["mstr"])}')
                # _ = self.fetcher_eagle.get_p_subhalo(idsubhalo=10)
            except Exception as e:
                print(e)
                continue

    def test_eagle_halo_no_min_mass(self):
        simulation_codes = ['RefL0025N0376',
                            'RefL0025N0752',
                            'RecalL0025N0752']

        for sim in simulation_codes:
            self.cfg_eagle.catalog['simulation_code'] = sim
            conn = get_connection()
            query = f"""
                        WITH RankedSubhalos AS (
                            SELECT 
                                SH.GroupID AS halo_id, -- Halo ID linked to the subhalo
                                SH.SubGroupNumber AS subhalo_id, -- Subhalo ID
                                SH.MassType_Star AS stellar_mass, -- Stellar mass of the subhalo
                                ROW_NUMBER() OVER(PARTITION BY SH.GroupID ORDER BY SH.MassType_Star DESC) AS rn -- Rank subhalos by mass
                            FROM
                                {sim}_Subhalo AS SH
                            WHERE
                                SH.SnapNum = {self.cfg_eagle.catalog.snapnum} -- Match the snapshot for consistency
                        )
                        SELECT
                            FOF.GroupID AS halo_id, -- Unique identifier for each halo
                            FOF.Group_M_Crit200 AS m200, -- Mass within R200 (critical density)
                            FOF.Group_R_Crit200 AS r200, -- Radius within which density is 200 times the critical density
                            FOF.GroupCentreOfPotential_x AS pos_x, -- X coordinate of halo's center
                            FOF.GroupCentreOfPotential_y AS pos_y, -- Y coordinate of halo's center
                            FOF.GroupCentreOfPotential_z AS pos_z, -- Z coordinate of halo's center
                            FOF.NumOfSubhalos AS num_subhalos, -- Number of subhalos in each halo
                            RS.subhalo_id AS max_subhalo_id, -- ID of the most massive subhalo in the halo
                            RS.stellar_mass AS max_subhalo_mass -- Stellar mass of the most massive subhalo
                        FROM
                            {sim}_FOF AS FOF -- Table for halo properties
                        LEFT JOIN 
                            RankedSubhalos AS RS ON FOF.GroupID = RS.halo_id AND RS.rn = 1 -- Only the most massive subhalo (rn = 1)
                        WHERE
                            FOF.SnapNum = {self.cfg_eagle.catalog.snapnum} -- Snapshot for redshift ~0.3
                        ORDER BY
                            FOF.GroupID ASC -- Order results by halo ID
                        """
            # Run the query
            array = sql.execute_query(conn, query)
            df = pd.DataFrame(array)
            print(len(df))

    def test_eagle_build_query(self):
        query = self.fetcher_eagle._build_query(table='subhalo')
        self.assertNotEqual(query, None)


