import os
import unittest

import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose
import eagleSqlTools as sql

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection

load_dotenv('../.env')


class TestFetcher(unittest.TestCase):
    def setUp(self):
        # Initialize Hydra and load configuration
        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")

        with initialize(config_path="../../../configs"):
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

        os.environ['PROJECT_ROOT'] = '../../..'

        self.fetcher_illustris = DataFetcherFactory.get_data_fetcher(self.cfg_illustris)
        self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(self.cfg_eagle)

    def test_get_halos_illustris(self):
        halo_context_1 = self.fetcher_illustris.get_halos()

        self.assertIsInstance(halo_context_1, HaloContext)
        self.assertTrue(
            halo_context_1.r200.shape[0] == halo_context_1.m200.shape[0] == halo_context_1.gpos.shape[0] ==
            halo_context_1.fsub.shape[0]
        )

        # get_item
        halo_data_1 = halo_context_1.get_item(halo_id=0)
        assert halo_data_1 is not None

    def test_get_halos_eagle(self):
        halo_context_2 = self.fetcher_eagle.get_halos()

        self.assertIsInstance(halo_context_2, HaloContext)
        self.assertTrue(
            halo_context_2.r200.shape[0] == halo_context_2.m200.shape[0] == halo_context_2.gpos.shape[0] ==
            halo_context_2.fsub.shape[0]
        )

        # get_item
        halo_data_2 = halo_context_2.get_item(halo_id=halo_context_2.halo_id[0])
        assert halo_data_2 is not None

    def test_get_subhalos_illustris(self):
        subhalo_context_1 = self.fetcher_illustris.get_subhalos()
        self.assertIsInstance(subhalo_context_1, SubhaloContext)
        self.assertTrue(
            subhalo_context_1.spos.shape[0] == subhalo_context_1.shmr.shape[0] ==
            subhalo_context_1.id_parents.shape[0] == subhalo_context_1.mstr.shape[0]
        )

        # Get item
        subhalo_data_1 = subhalo_context_1.get_item(subhalo_id=0)
        assert subhalo_data_1 is not None

    def test_get_subhalos_eagle(self):
        subhalo_context_2 = self.fetcher_eagle.get_subhalos()
        self.assertIsInstance(subhalo_context_2, SubhaloContext)
        self.assertTrue(
            subhalo_context_2.spos.shape[0] == subhalo_context_2.shmr.shape[0] ==
            subhalo_context_2.id_parents.shape[0] == subhalo_context_2.mstr.shape[0]
        )

        # Get item
        subhalo_data_2 = subhalo_context_2.get_item(subhalo_id=subhalo_context_2.subhalo_id[0])
        assert subhalo_data_2 is not None

    def test_get_p_subhalo_illustris(self):
        subhalo_particles_1 = self.fetcher_illustris.get_p_subhalo(idsubhalo=10)
        self.assertIsInstance(subhalo_particles_1, SubhaloParticles)
        self.assertTrue(
            subhalo_particles_1.p.shape[0] == subhalo_particles_1.m.shape[0] == subhalo_particles_1.v.shape[0]
        )
        self.assertTrue(
            subhalo_particles_1.p.shape[1] == subhalo_particles_1.v.shape[1]
        )

    def test_get_p_subhalo_eagle(self):
        self.cfg_eagle.catalog.simulation_code = 'RefL0025N0376'
        self.cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0025N0376'
        subhalo_particles_2 = self.fetcher_eagle.get_p_subhalo(subgroup_id=0, halo_id=620)
        self.assertIsInstance(subhalo_particles_2, SubhaloParticles)
        self.assertTrue(
            subhalo_particles_2.p.shape[0] == subhalo_particles_2.m.shape[0] == subhalo_particles_2.v.shape[0]
        )
        self.assertTrue(
            subhalo_particles_2.p.shape[1] == subhalo_particles_2.v.shape[1]
        )

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
                print(f'-- Number of halos: {len(halos.m200)}')
                subhalos = self.fetcher_eagle.get_subhalos()
                print(f'-- Number of subhalos: {len(subhalos.mstr)}')
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

