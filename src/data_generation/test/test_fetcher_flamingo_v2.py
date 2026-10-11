import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

import rootutils
from src.data_generation.src.data_fetcher import FlamingoFetcher, FlamingoFetcherV2
from src.data_generation.src.subhalo import SubhaloParticles

rootutils.setup_root(__file__, pythonpath=True)

# Reuse the exact fixtures/helpers already used to test FlamingoFetcher (the
# FakeSwiftField mock, catalog builders, real-data helpers), so V2's tests
# stay faithful to the real SOAP/SWIFTDataset object shape rather than
# inventing a parallel, possibly-incompatible mock.
from src.data_generation.test.test_fetcher_flamingo import (
    FakeSwiftField,
    make_swift_field,
    make_fake_swift_dataset,
    make_fake_particles,
    make_fake_processed_particles,
    make_mock_catalog,
    make_real_catalog,
    _check_real_data_exists,
    PIPELINE_PATH,
    USE_REAL_DATA,
)


@pytest.fixture
def mock_catalog():
    return make_mock_catalog()


@pytest.fixture
def fetcher(mock_catalog):
    fof = make_fake_swift_dataset(n=10)
    parts = make_fake_particles(n=100)

    with patch('os.path.exists', return_value=True), \
         patch('swiftsimio.load', side_effect=[fof, parts]):
        f = FlamingoFetcherV2(catalog=mock_catalog)

    f.fof_subhalos = fof
    f.particles = parts
    f.halo_fields = [
        'input_halos.halo_centre',
        'spherical_overdensity_200_crit.total_mass',
        'spherical_overdensity_200_crit.r_200_crit',
    ]
    f.subhalo_fields = ['input_halos.halo_centre']
    f.particle_fields = ['stars']
    f.column_map = {
        'input_halos.halo_centre':                   'gpos',
        'spherical_overdensity_200_crit.total_mass': 'm200',
        'spherical_overdensity_200_crit.r_200_crit': 'r200',
        'halo_id':                                   'halo_id',
        'fsub':                                      'fsub',
        'subhalo_id':                                'subhalo_id',
        'id_parents':                                'id_parents',
        'soap.host_halo_index':                      'soap.host_halo_index',
        'spos':                                      'spos',
    }
    return f


@pytest.fixture
def mock_halo():
    halo = MagicMock()
    halo.halo_id = 0
    halo.is_primary = -1
    halo.gpos = np.array([250.0, 250.0, 250.0])
    return halo


def _capture_primary_mask(fetcher, call):
    """Run `call` (a no-arg callable invoking the fetcher) with
    _validate_and_process_fields replaced by a spy, and return the boolean
    mask it was invoked with for the 'stars' particle type — i.e. exactly
    the set of particles the fetcher decided belong to the halo/subhalo,
    without depending on the rest of the (mocked-elsewhere) pipeline."""
    captured = {}

    def fake_validate(particles, fields, super_key, mask=None, n_rows=None):
        captured['mask'] = mask
        return make_fake_processed_particles()

    with patch.object(fetcher, '_validate_and_process_fields', side_effect=fake_validate), \
         patch(PIPELINE_PATH, return_value=make_fake_processed_particles()):
        call()

    return captured['mask']['stars']


class TestFlamingoFetcherV2:

    # ── basic behaviour is inherited unchanged from FlamingoFetcher ───────────

    def test_get_p_halo_returns_subhalo_particles(self, fetcher, mock_halo):
        with patch(PIPELINE_PATH, return_value=make_fake_processed_particles()):
            result = fetcher.get_p_halo(halo=mock_halo)
        assert isinstance(result, SubhaloParticles)

    def test_get_p_halo_raises_if_not_primary(self, fetcher, mock_halo):
        mock_halo.is_primary = 1
        with pytest.raises(ValueError, match="not a primary subhalo"):
            fetcher.get_p_halo(halo=mock_halo)

    def test_get_p_halo_raises_if_halo_missing(self, fetcher):
        with pytest.raises(ValueError, match="halo must be provided"):
            fetcher.get_p_halo()

    def test_get_p_halo_raises_if_out_of_range(self, fetcher, mock_halo):
        """Regression: _get_p_halo must bounds-check halo.halo_id before
        indexing input_halos.halo_catalogue_index with it, the same way
        _get_p_subhalo already bounds-checks subhalo_id — an out-of-range id
        used to fall through to a raw IndexError instead of this ValueError."""
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        mock_halo.halo_id = 999
        with pytest.raises(ValueError, match="not a primary subhalo"):
            fetcher.get_p_halo(halo=mock_halo)

    def test_get_p_subhalo_returns_subhalo_particles(self, fetcher):
        fetcher.fof_subhalos.soap.host_halo_index = make_swift_field(np.array([-1] * 10))
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        with patch(PIPELINE_PATH, return_value=make_fake_processed_particles()):
            result = fetcher.get_p_subhalo(subhalo_id=0)
        assert isinstance(result, SubhaloParticles)

    def test_get_p_subhalo_raises_if_id_missing(self, fetcher):
        with pytest.raises(ValueError, match="subhalo_id must be provided"):
            fetcher.get_p_subhalo()

    def test_get_p_subhalo_raises_if_not_primary(self, fetcher):
        fetcher.fof_subhalos.soap.host_halo_index = make_swift_field(np.ones(10, dtype=int))
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        with pytest.raises(ValueError, match="not a primary subhalo"):
            fetcher.get_p_subhalo(subhalo_id=0)

    def test_get_p_subhalo_raises_if_out_of_range(self, fetcher):
        """V2 bounds-checks the row-position id directly instead of V1's
        'search for a row, fail to find one' path — same observable error."""
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        with pytest.raises(ValueError, match="not a primary subhalo"):
            fetcher.get_p_subhalo(subhalo_id=999)

    def test_get_p_halo_includes_satellite_particles(self, fetcher, mock_halo):
        """Satellite-gathering logic is untouched (row-position throughout,
        confirmed correct by the SOAP docs: HostHaloIndex is documented as
        'index within the SOAP arrays') — must still work under V2."""
        fetcher.fof_subhalos.soap.host_halo_index = make_swift_field(
            np.array([-1, -1, 0, 0, -1, -1, -1, -1, -1, -1])
        )
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        fetcher.particles.stars.halo_catalogue_index = make_swift_field(
            np.repeat(np.arange(10), 10)
        )
        with patch(PIPELINE_PATH, return_value=make_fake_processed_particles(n=30)):
            result = fetcher.get_p_halo(halo=mock_halo)
        assert isinstance(result, SubhaloParticles)

    # ── regression: V2 must use true HaloCatalogueIndex, not row-position ─────
    # (this is the bug itself, reproduced directly: row-position 0's TRUE
    # catalogue index is 5, and 10 particles are deliberately tagged with
    # each of halo_catalogue_index==0 (the row-position decoy) and ==5 (the
    # true owner) so a wrong implementation and a correct one disagree.)

    def _make_diverging_dataset(self, fetcher):
        # row i's true SOAP catalogue index is diverging_ids[i]; row 0 -> 5.
        diverging_ids = np.array([5, 3, 8, 1, 9, 0, 7, 2, 6, 4])
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(diverging_ids)
        fetcher.fof_subhalos.soap.host_halo_index = make_swift_field(np.array([-1] * 10))
        # particles 0-9 tagged halo_catalogue_index=0, 10-19 tagged =1, ... 90-99 tagged =9
        fetcher.particles.stars.halo_catalogue_index = make_swift_field(np.repeat(np.arange(10), 10))
        return diverging_ids

    def test_get_p_halo_uses_true_catalogue_index_not_row_position(self, fetcher, mock_halo):
        self._make_diverging_dataset(fetcher)
        mock_halo.halo_id = 0  # row-position; true catalogue index is 5

        primary_mask = _capture_primary_mask(fetcher, lambda: fetcher.get_p_halo(halo=mock_halo))

        # Particles tagged with the TRUE catalogue index (5) must be selected...
        assert primary_mask[50:60].all(), "particles truly belonging to this halo were dropped"
        # ...and particles tagged with the row-position decoy (0) must NOT be
        # selected — V1's bug was matching exactly this wrong group.
        assert not primary_mask[0:10].any(), "V2 reproduced V1's row-position false positive"

    def test_get_p_subhalo_uses_true_catalogue_index_not_row_position(self, fetcher):
        self._make_diverging_dataset(fetcher)

        primary_mask = _capture_primary_mask(
            fetcher, lambda: fetcher.get_p_subhalo(subhalo_id=0)  # true catalogue index is 5
        )

        assert primary_mask[50:60].all(), "particles truly belonging to this subhalo were dropped"
        assert not primary_mask[0:10].any(), "V2 reproduced V1's row-position false positive"

    def test_v1_reproduces_the_false_positive_v2_fixes(self, fetcher, mock_halo):
        """Direct side-by-side: same diverging dataset, V1's unmodified logic
        actually exercised (not just asserted about) to prove the contrast is
        real, not assumed."""
        self._make_diverging_dataset(fetcher)
        mock_halo.halo_id = 0

        v1_primary_mask = _capture_primary_mask(fetcher, lambda: FlamingoFetcher._get_p_halo(fetcher, mock_halo))
        assert v1_primary_mask[0:10].all(), "sanity check: V1 should match the row-position decoy group"
        assert not v1_primary_mask[50:60].any(), "sanity check: V1 should miss the true owner group"

        v2_primary_mask = _capture_primary_mask(fetcher, lambda: fetcher.get_p_halo(halo=mock_halo))
        assert v2_primary_mask[50:60].all()
        assert not v2_primary_mask[0:10].any()


@pytest.mark.skipif(not USE_REAL_DATA, reason="USE_REAL_DATA=0")
class TestFlamingoFetcherV2RealData:
    """
    Integration tests against real Flamingo data, mirroring
    TestFlamingoFetcherRealData in test_fetcher_flamingo.py.

    Run with:
        USE_REAL_DATA=1 pytest tests/test_fetcher_flamingo_v2.py -v -k RealData
    """

    @pytest.fixture
    def real_fetcher_v1(self):
        ok, reason = _check_real_data_exists()
        if not ok:
            pytest.skip(reason)
        config = make_real_catalog()
        required_keys = ['G']
        data_loader_dict = next((step for step in config.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        return FlamingoFetcher(catalog=config.catalog, **attr_dict)

    @pytest.fixture
    def real_fetcher_v2(self):
        ok, reason = _check_real_data_exists()
        if not ok:
            pytest.skip(reason)
        config = make_real_catalog()
        required_keys = ['G']
        data_loader_dict = next((step for step in config.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        return FlamingoFetcherV2(catalog=config.catalog, **attr_dict)

    def _first_diverging_halo(self, fetcher, halos):
        """Find a halo whose row-position id differs from its true SOAP
        catalogue index — i.e. one where V1's bug actually bites — rather
        than assuming the first halo happens to be such a case."""
        for halo_id in halos.data['halo_id'].tolist():
            true_catalogue_id = int(fetcher.fof_subhalos.input_halos.halo_catalogue_index[halo_id])
            if true_catalogue_id != halo_id:
                return halo_id, true_catalogue_id
        return None, None

    def test_v2_matches_true_catalogue_index_ground_truth(self, real_fetcher_v2):
        """
        Ground-truth check independent of both fetchers: directly count how
        many real star particles carry halo_catalogue_index == the halo's
        TRUE SOAP catalogue index, and confirm V2 returns exactly that many
        (not the row-position count V1 would return).
        """
        halos = real_fetcher_v2.get_halos()
        halo_id, true_catalogue_id = self._first_diverging_halo(real_fetcher_v2, halos)
        if halo_id is None:
            pytest.skip("No halo in this fixture has row-position != true catalogue index")

        halo = halos.get_halo_by_id(halo_id)
        raw_ids = np.asarray(real_fetcher_v2.particles.stars.halo_catalogue_index)
        expected_n = int((raw_ids == true_catalogue_id).sum())

        result = real_fetcher_v2.get_p_halo(halo=halo)
        assert len(result.star_p) == expected_n

    def test_v1_and_v2_disagree_when_row_position_diverges(self, real_fetcher_v1, real_fetcher_v2):
        """
        The actual bug and fix, exercised end-to-end on real production-scale
        data (not a mock): for a halo where row-position != true catalogue
        index, V1 and V2 must select different particle counts, and V2's
        count must match the row's true catalogue index, not its row
        position.
        """
        halos = real_fetcher_v1.get_halos()
        halo_id, true_catalogue_id = self._first_diverging_halo(real_fetcher_v1, halos)
        if halo_id is None:
            pytest.skip("No halo in this fixture has row-position != true catalogue index")

        halo = halos.get_halo_by_id(halo_id)
        raw_ids = np.asarray(real_fetcher_v1.particles.stars.halo_catalogue_index)
        n_matching_row_position = int((raw_ids == halo_id).sum())
        n_matching_true_id = int((raw_ids == true_catalogue_id).sum())

        result_v1 = real_fetcher_v1.get_p_halo(halo=halo)
        result_v2 = real_fetcher_v2.get_p_halo(halo=halo)

        assert len(result_v1.star_p) == n_matching_row_position
        assert len(result_v2.star_p) == n_matching_true_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
