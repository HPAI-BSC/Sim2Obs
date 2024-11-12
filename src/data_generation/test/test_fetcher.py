import unittest
from dotenv import load_dotenv

from src.data_generation.src.constants import PATH
from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles

load_dotenv('../.env')


class TestFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher_illustris = DataFetcherFactory.get_data_fetcher(catalog="illustris-300", arg=PATH)
        self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(catalog="eagle", arg="RefL0100N1504")

    def test_get_halos(self):
        halo_context_1 = self.fetcher_illustris.get_halos()
        halo_context_2 = self.fetcher_eagle.get_halos()

        self.assertIsInstance(halo_context_1, HaloContext)
        self.assertTrue(
            halo_context_1.r200.shape[0] == halo_context_1.m200.shape[0] == halo_context_1.gpos.shape[0] ==
            halo_context_1.fsub.shape[0]
        )
        self.assertIsInstance(halo_context_2, HaloContext)
        self.assertTrue(
            halo_context_2.r200.shape[0] == halo_context_2.m200.shape[0] == halo_context_2.gpos.shape[0] ==
            halo_context_2.fsub.shape[0]
        )

    def test_get_subhalos(self):
        subhalo_context_1 = self.fetcher_illustris.get_subhalos()
        subhalo_context_2 = self.fetcher_eagle.get_subhalos()
        self.assertIsInstance(subhalo_context_1, SubhaloContext)
        self.assertTrue(
            subhalo_context_1.spos.shape[0] == subhalo_context_1.shmr.shape[0] ==
            subhalo_context_1.id_parents.shape[0] == subhalo_context_1.mstr.shape[0]
        )
        self.assertIsInstance(subhalo_context_2, SubhaloContext)
        self.assertTrue(
            subhalo_context_2.spos.shape[0] == subhalo_context_2.shmr.shape[0] ==
            subhalo_context_2.id_parents.shape[0] == subhalo_context_2.mstr.shape[0]
        )

    def test_get_p_subhalo(self):
        subhalo_particles_1 = self.fetcher_illustris.get_p_subhalo(idsubhalo=10)
        subhalo_particles_2 = self.fetcher_eagle.get_p_subhalo(idsubhalo=10)
        self.assertIsInstance(subhalo_particles_1, SubhaloParticles)
        self.assertTrue(
            subhalo_particles_1.p.shape[0] == subhalo_particles_1.m.shape[0] == subhalo_particles_1.v.shape[0]
        )
        self.assertTrue(
            subhalo_particles_1.p.shape[1] == subhalo_particles_1.v.shape[1]
        )
        self.assertIsInstance(subhalo_particles_2, SubhaloParticles)
        self.assertTrue(
            subhalo_particles_2.p.shape[0] == subhalo_particles_2.m.shape[0] == subhalo_particles_2.v.shape[0]
        )
        self.assertTrue(
            subhalo_particles_2.p.shape[1] == subhalo_particles_2.v.shape[1]
        )
