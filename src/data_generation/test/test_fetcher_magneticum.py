import os
import unittest

import g3read
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import numpy as np
from dotenv import load_dotenv
from hydra import initialize, compose

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles

load_dotenv('../.env')


class TestFetcherMagneticum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['PROJECT_ROOT'] = '../../..'

        # Initialize Hydra and load configuration
        with initialize(config_path="../../../configs"):
            cls.cfg_magneticum = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            cls.cfg_magneticum.paths = local_config.paths
            cls.cfg_magneticum.catalog = compose("catalog/magneticum.yaml")['catalog']
            cls.cfg_magneticum.catalog.path = 'data/magneticum'

        required_keys = ['G']
        data_loader_dict = next(
            (step for step in cls.cfg_magneticum.pipeline if step.name == 'data_loader'), None
        )
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        cls.fetcher_magneticum = DataFetcherFactory.get_data_fetcher(cls.cfg_magneticum.catalog, **attr_dict)

    def test_get_halos(self):
        halo_context = self.fetcher_magneticum.get_halos()
        self.assertIsInstance(halo_context, HaloContext)
        self.assertIsNotNone(halo_context.data)
        self.assertGreater(len(halo_context.data), 0)
        self.assertIn('halo_id', halo_context.data.columns)
        self.assertIn('subhalo_ids', halo_context.data.columns)  # vectorized children

    def test_get_halos_subhalo_ids_are_arrays(self):
        """subhalo_ids should contain numpy arrays of indices."""
        halo_context = self.fetcher_magneticum.get_halos()
        first_subhalo_ids = halo_context.data['subhalo_ids'].iloc[0]
        self.assertIsInstance(first_subhalo_ids, np.ndarray)

    def test_get_halos_subhalo_ids_correct_count(self):
        """Number of subhalo ids should match NSUB for each halo."""
        halo_context = self.fetcher_magneticum.get_halos()
        for _, row in halo_context.data.iterrows():
            self.assertEqual(
                len(row['subhalo_ids']),
                self.fetcher_magneticum._nsub[row['halo_id']],
                msg=f"Halo {row['halo_id']} has wrong number of subhalo ids"
            )

    # Subhalos
    def test_debug_subhalo_arrays(self):
        """Check array lengths before pipeline."""
        if not getattr(self.fetcher_magneticum, '_cache_loaded', False):
            self.fetcher_magneticum._load_catalog_cache()

        group_base = self.fetcher_magneticum._group_base
        subhalo_fields = self.fetcher_magneticum.subhalo_fields.copy()
        subhalo_fields.remove('stellar_half_mass_radius')
        subhalos = g3read.read_new(group_base, subhalo_fields, 1, is_snap=False, multiple_files=True)

        for key, val in subhalos.items():
            print(f"{key}: {len(val) if hasattr(val, '__len__') else val}")

    def test_get_subhalos(self):
        subhalo_context = self.fetcher_magneticum.get_subhalos()
        self.assertIsInstance(subhalo_context, SubhaloContext)
        self.assertIsNotNone(subhalo_context.data)
        self.assertGreater(len(subhalo_context.data), 0)
        self.assertIn('subhalo_id', subhalo_context.data.columns)

    def test_subhalo_belongs_to_halo(self):
        """Sanity check: subhalo 0 should belong to halo 0."""
        subhalo_context = self.fetcher_magneticum.get_subhalos()
        subhalo = subhalo_context.get_subhalo_by_id(subhalo_id=0)
        self.assertEqual(subhalo.id_parents, 0)

    # Particles subhalo
    def test_get_p_subhalo_returns_instance(self):
        particles = self.fetcher_magneticum.get_p_subhalo(subhalo_id=0)
        self.assertIsInstance(particles, SubhaloParticles)

    def test_get_p_subhalo_central_has_stars(self):
        """Central subhalo (id=0) should have stellar particles."""
        particles = self.fetcher_magneticum.get_p_subhalo(subhalo_id=0)
        self.assertFalse(particles.is_empty)
        self.assertIsNotNone(particles.star_p)
        self.assertGreater(len(particles.star_p), 0)

    def test_get_p_subhalo_empty_returns_empty(self):
        """A subhalo known to be empty should return empty SubhaloParticles."""
        particles = self.fetcher_magneticum.get_p_subhalo(subhalo_id=20308)
        self.assertTrue(particles.is_empty)
        self.assertIsInstance(particles, SubhaloParticles)

    def test_get_p_subhalo_uses_cache(self):
        """Calling get_p_subhalo twice should use cache — _cache_loaded stays True."""
        self.fetcher_magneticum.get_p_subhalo(subhalo_id=0)
        self.fetcher_magneticum.get_p_subhalo(subhalo_id=0)
        self.assertTrue(getattr(self.fetcher_magneticum, '_cache_loaded', False))

    # ── Particles: halo ──────────────────────────────────────────────────────

    def test_get_p_halo_returns_instance(self):
        halo = Halo(halo_id=0, fsub=0)
        particles = self.fetcher_magneticum.get_p_halo(halo=halo)
        self.assertIsInstance(particles, SubhaloParticles)

    def test_get_p_halo_has_stars(self):
        halo = Halo(halo_id=0, fsub=0)
        particles = self.fetcher_magneticum.get_p_halo(halo=halo)
        self.assertIsNotNone(particles.star_p)
        self.assertGreater(len(particles.star_p), 0)

    def test_halo_particle_count_gte_subhalo(self):
        """Halo 0 should have >= stellar particles than its central subhalo."""
        halo = Halo(halo_id=0, fsub=0)
        halo_particles = self.fetcher_magneticum.get_p_halo(halo=halo)
        subhalo_particles = self.fetcher_magneticum.get_p_subhalo(subhalo_id=0)
        self.assertGreaterEqual(
            len(halo_particles.star_p),
            len(subhalo_particles.star_p),
            msg="Halo should have >= stars than its central subhalo"
        )

    # ── All subhalo particles ────────────────────────────────────────────────
    def test_get_all_subhalo_particles_returns_dict(self):
        all_particles = self.fetcher_magneticum._load_subhalo_particles_cache()
        self.assertIsInstance(self.fetcher_magneticum.all_particles, dict)
        self.assertGreater(len(self.fetcher_magneticum.all_particles), 0)

    def test_get_all_subhalo_particles_first_subhalo_exists(self):
        if not getattr(self.fetcher_magneticum, '_subhalo_particles_cache_loaded', False):
            self.fetcher_magneticum._load_subhalo_particles_cache()
        self.assertIn(0, self.fetcher_magneticum.all_particles)

    def test_get_all_subhalo_particles_values_are_dicts(self):
        """Raw cache should contain dicts, not SubhaloParticles."""
        if not getattr(self.fetcher_magneticum, '_subhalo_particles_cache_loaded', False):
            self.fetcher_magneticum._load_subhalo_particles_cache()
        for sub_idx, data in self.fetcher_magneticum.all_particles.items():
            self.assertIsInstance(data, dict,
                                  msg=f"Subhalo {sub_idx} should be raw dict in cache")

    def test_get_all_subhalo_particles_central_largest(self):
        """Central subhalo (0) should have >= particles than last subhalo."""
        if not getattr(self.fetcher_magneticum, '_subhalo_particles_cache_loaded', False):
            self.fetcher_magneticum._load_subhalo_particles_cache()
        last_idx = max(self.fetcher_magneticum.all_particles.keys())
        first_n_stars = len(self.fetcher_magneticum.all_particles[0].get('star', {}).get('POS', []))
        last_n_stars = len(self.fetcher_magneticum.all_particles[last_idx].get('star', {}).get('POS', []))
        self.assertGreaterEqual(first_n_stars, last_n_stars)

    # ── Cache ────────────────────────────────────────────────────────────────

    def test_cache_loaded_after_first_call(self):
        self.fetcher_magneticum.get_halos()
        self.assertTrue(getattr(self.fetcher_magneticum, '_cache_loaded', False))

    def test_sorted_snap_cache_built(self):
        """_sorted_snap_cache should be built after _load_catalog_cache."""
        if not getattr(self.fetcher_magneticum, '_cache_loaded', False):
            self.fetcher_magneticum._load_catalog_cache()
        self.assertTrue(hasattr(self.fetcher_magneticum, '_sorted_snap_cache'))
        for part_type in self.fetcher_magneticum._snap_cache:
            self.assertIn(part_type, self.fetcher_magneticum._sorted_snap_cache)
            self.assertIn('sorted_idx', self.fetcher_magneticum._sorted_snap_cache[part_type])
            self.assertIn('sorted_ids', self.fetcher_magneticum._sorted_snap_cache[part_type])

    def test_particle_conditions_cached(self):
        """_particle_conditions should be pre-computed after cache load."""
        if not getattr(self.fetcher_magneticum, '_cache_loaded', False):
            self.fetcher_magneticum._load_catalog_cache()
        self.assertTrue(hasattr(self.fetcher_magneticum, '_particle_conditions'))
        self.assertIsInstance(self.fetcher_magneticum._particle_conditions, dict)