"""
Smoke test of the use case "dark matter mass map from the stellar metallicity map" on REAL data.

Not a pytest fixture, run it directly on MN5 (it needs the snapshots and the `*_dm_v1` dataset directories with
their `output` symlinks):

    python src/data_generation/test/manual_tests/manual_test_dm_use_case.py

The parameters are hardcoded below. It composes configs/data_generation_dm.yaml for each catalog variant, builds
the real fetcher and, for a few halos, checks the whole chain before a long run:

  Directory   the dataset path and its `output` symlink resolve, MassTable[1] is readable
  Particles   one halo is loaded through the real fetcher: classes, attributes, counts, dtypes, no NaN, wind
              particles removed, metallicity range, constant DM mass equal to MassTable[1] * 1e10
  Catalogue   the particles against the group catalogue: number of DM particles equals SubhaloLenType (exact),
              summed DM mass equals SubhaloMassType[:, 1], stars not above SubhaloMassType[:, 4], and how much of
              the FoF group's DM the loaded subhalos hold (the subhalo filters of the YAML and the unbound
              material leave part of it out, which decides what the DM target map really measures)
  Geometry    positions after recentring: star and DM centres agree and sit on the halo centre, radii are of
              the order of R200, DM mass inside R200 is a sensible fraction of M200 (a joint check of the
              position scaling, the mass unit and R200)
  Maps        the two features on the real particles, 3 projections: shapes, finite, DM map equals an independent
              histogram, share of DM mass inside the map window, empty pixels, metallicity range
  Neighbour   a subhalo of another halo is loaded as the data loader does: stars only, and the per-type overlap
              boundaries recover the halo's own particles
  Rotation    SubhaloParticlesV2.rotate refuses several particle types, instead of silently misaligning them
  Output      if maps were already generated (halos 0..3 of the pipeline), the folder layout, file names,
              shapes, value ranges, and what the interloper mask channel contains

It also reports the load time and the peak memory of each halo. Dark matter makes the loading much heavier than
the stellar-only runs, this is the number to look at before launching the full job.

Exit code is 1 if any FAIL is reported.
"""
import json
import os
import platform
import re
import resource
import sys
import time

import h5py
import illustris_python as il
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hydra import compose, initialize_config_dir  # noqa: E402

from src.data_generation.src.data_loader_step import DataLoader  # noqa: E402
from src.data_generation.src.feature import FeatureProcessorFactory  # noqa: E402
from src.data_generation.src.subhalo_particles_v2 import SubhaloParticlesV2  # noqa: E402
from src.data_generation.src.utils import muse_bins_r  # noqa: E402

CATALOGS = ['millenium_dm'] #'illustris300_dm'
# Positions in the halo list of the pipeline (catalogue order, the first ones are the most massive halos and
# so the heaviest to load). Positions beyond the end of the list are ignored.
HALO_POSITIONS = [0, 200]
DATASETS_DIR = None            # None: the one of configs/paths/gpfs.yaml
MAX_SUBHALOS_FOR_MASS_CHECK = 3000      # loading one catalogue record per subhalo, skip the check above this
MAX_OUTPUT_FILES_TO_READ = 60
JSON_OUTPUT = None             # e.g. 'dm_smoke_report.json' to also save the report

MASS_UNIT = 1e10               # the catalogs convert masses with 'x * 1e10', 10^10 Msun/h -> Msun/h
PROJECTIONS = {'xy': (0, 1), 'yz': (1, 2), 'xz': (0, 2)}
REQUIRED_ATTRIBUTES = ['star_p', 'star_v', 'star_m', 'star_metallicity', 'star_formation_time', 'dm_p', 'dm_m']
FILE_PATTERN = re.compile(r'halo_(\d+)_subhalo_(\d+)_(xy|yz|xz)_zoom_(\d+)\.npz')

DM_MASS_REL_TOL = 1e-4         # summed float32 catalogue masses against the float64 particle masses
# Geometry thresholds. A unit or scaling error is off by a factor of 10 or more, these are loose on purpose
CENTRE_OFFSET_WARN = 0.5       # in units of R200
RADIUS_FAIL_RANGE = (0.01, 10.0)    # median particle radius in units of R200
DM_FRACTION_OF_M200_WARN = (0.3, 1.0)   # DM mass inside R200 over M200, about 0.84 for a complete halo


class Report:
    """Collects (status, label, detail) lines of one catalog and prints them as it goes."""

    def __init__(self, name):
        self.name = name
        self.lines = []

    def add(self, status, label, detail=''):
        self.lines.append({'status': status, 'label': label, 'detail': detail})
        print(f'  [{status:<4}] {label}' + (f': {detail}' if detail else ''))

    @property
    def failed(self):
        return any(line['status'] == 'FAIL' for line in self.lines)


def peak_rss_mb():
    """Peak resident memory of this process in MB (ru_maxrss is bytes on macOS and KB on Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024 if platform.system() == 'Darwin' else 1024)


def current_rss_mb():
    """Resident memory of this process right now in MB (Linux), the peak when /proc is not available."""
    try:
        with open('/proc/self/statm') as statm:
            return int(statm.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1024 ** 2
    except OSError:
        return peak_rss_mb()


def ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float('nan')


class CatalogLookup:
    """
    Group catalogue records of single subhalos or halos, reading ONLY the fields the checks need.

    `il.groupcat.loadSingle` reads every field of the record, which fails on the Millennium-TNG catalogue (it has a
    field with a 6 byte integer type that h5py cannot read). This follows the same steps (offsets file, then the
    chunk file that holds the record) for the requested fields only, and keeps the last chunk file open because the
    subhalos of a halo are consecutive.
    """
    FIELDS = {'Subhalo': ['SubhaloLenType', 'SubhaloMassType'], 'Group': ['GroupMassType']}

    def __init__(self, base_path, snapnum):
        self.base_path, self.snapnum = base_path, snapnum
        self._offsets = {}
        self._open_file = None
        self._open_chunk = None

    def _file_offsets(self, name):
        if name not in self._offsets:
            if 'fof_subhalo' in il.groupcat.gcPath(self.base_path, self.snapnum):
                with h5py.File(il.groupcat.offsetPath(self.base_path, self.snapnum), 'r') as offsets_file:
                    self._offsets[name] = offsets_file['FileOffsets/' + name][()]
            else:
                with h5py.File(il.groupcat.gcPath(self.base_path, self.snapnum), 'r') as first_chunk:
                    self._offsets[name] = first_chunk['Header'].attrs['FileOffsets_' + name]
        return self._offsets[name]

    def _record(self, name, index):
        relative = int(index) - self._file_offsets(name)
        chunk = int(np.max(np.where(relative >= 0)))
        if self._open_chunk != chunk:
            self.close()
            self._open_file = h5py.File(il.groupcat.gcPath(self.base_path, self.snapnum, chunk), 'r')
            self._open_chunk = chunk
        local = int(relative[chunk])
        return {field: self._open_file[name][field][local] for field in self.FIELDS[name]}

    def subhalo(self, subhalo_id):
        return self._record('Subhalo', subhalo_id)

    def group(self, halo_id):
        return self._record('Group', halo_id)

    def close(self):
        if self._open_file is not None:
            self._open_file.close()
        self._open_file, self._open_chunk = None, None


def guarded(rep, label, function, *args):
    """Run one check section. An exception becomes a FAIL line and the remaining sections still run."""
    try:
        return function(*args)
    except Exception as error:      # noqa: BLE001
        rep.add('FAIL', f'{label} raised an exception', f'{type(error).__name__}: {error}')
        return None


def load_config(catalog):
    overrides = [f'catalog={catalog}'] + ([f'paths.datasets_dir={DATASETS_DIR}'] if DATASETS_DIR else [])
    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, 'configs'), version_base=None):
        return compose(config_name='data_generation_dm.yaml', overrides=overrides)


# ---------------------------------------------------------------------------------------------- directory

def check_dataset_directory(rep, cfg, fetcher):
    catalog = cfg.catalog
    rep.add('INFO', 'catalog', f'{catalog.name}, snapnum {catalog.snapnum}, a={catalog.a}, path={catalog.path}')
    output = os.path.join(catalog.path, 'output')
    if not os.path.isdir(output):
        rep.add('FAIL', 'dataset output directory', f'{output} does not exist, create the symlink with ln -s')
        return False
    rep.add('OK', 'dataset output directory', f'{output} -> {os.path.realpath(output)}')
    try:
        mass_table = fetcher._get_mass_table()
    except Exception as error:      # noqa: BLE001
        rep.add('FAIL', 'snapshot header', f'cannot read MassTable: {error}')
        return False
    rep.add('OK' if mass_table[1] > 0 else 'FAIL', 'MassTable[1] (DM mass, 1e10 Msun/h)', f'{mass_table[1]:.6g}')
    return mass_table[1] > 0


# ---------------------------------------------------------------------------------------------- particles

def check_particles(rep, own, mass_table):
    rep.add('OK' if isinstance(own, SubhaloParticlesV2) else 'FAIL', 'particle class', type(own).__name__)
    missing = [attribute for attribute in REQUIRED_ATTRIBUTES if not hasattr(own, attribute)]
    if missing:
        rep.add('FAIL', 'attributes', f'missing {missing}')
        return False
    rep.add('INFO', 'particles', f'stars {len(own.star_m)}, dark matter {len(own.dm_m)}')
    rep.add('INFO', 'dtypes', ', '.join(f'{a}={getattr(own, a).dtype}' for a in REQUIRED_ATTRIBUTES))

    nonfinite = [a for a in REQUIRED_ATTRIBUTES if not np.isfinite(np.asarray(getattr(own, a), dtype=float)).all()]
    rep.add('FAIL' if nonfinite else 'OK', 'no NaN or inf in any attribute', str(nonfinite) if nonfinite else '')

    wind = int((own.star_formation_time <= 0).sum())
    rep.add('FAIL' if wind else 'OK', 'wind particles removed (formation time > 0 for every star)', f'{wind} left' if wind else '')

    z = np.asarray(own.star_metallicity, dtype=float)
    z_mean = float(np.average(z, weights=own.star_m))
    rep.add('INFO', 'stellar metallicity', f'min={z.min():.4g}, mass-weighted mean={z_mean:.4g}, max={z.max():.4g}')
    rep.add('OK' if z.max() <= 1 and z.min() > -1e-3 else 'FAIL', 'metallicity looks like a mass fraction in [0, 1]')
    if z.min() < 0:
        rep.add('INFO', 'negative metallicities', f'{int((z < 0).sum())}, clipped to 0 by clip_min in the feature')

    expected = mass_table[1] * MASS_UNIT
    unique = np.unique(own.dm_m)
    ok = len(unique) == 1 and np.isclose(unique[0], expected, rtol=1e-6)
    rep.add('OK' if ok else 'FAIL', 'DM mass is the constant MassTable[1] * 1e10 Msun/h',
            f'{unique[:3]} (expected {expected:.6g})' if not ok else f'{expected:.6g} Msun/h')
    rep.add('OK' if (own.star_m > 0).all() else 'FAIL', 'star masses are positive')
    return not nonfinite and not wind


# ---------------------------------------------------------------------------------------------- catalogue

def check_catalogue(rep, own, halo, lookup):
    subhalo_ids = [int(i) for i in halo.subhalo_ids]
    if len(subhalo_ids) > MAX_SUBHALOS_FOR_MASS_CHECK:
        rep.add('INFO', 'catalogue mass check skipped', f'{len(subhalo_ids)} subhalos, above {MAX_SUBHALOS_FOR_MASS_CHECK}')
        return
    totals = {'dm_len': 0, 'dm_mass': 0.0, 'star_len': 0, 'star_mass': 0.0}
    for subhalo_id in subhalo_ids:
        record = lookup.subhalo(subhalo_id)
        totals['dm_len'] += int(record['SubhaloLenType'][1])
        totals['dm_mass'] += float(record['SubhaloMassType'][1])
        totals['star_len'] += int(record['SubhaloLenType'][4])
        totals['star_mass'] += float(record['SubhaloMassType'][4])

    rep.add('INFO', 'subhalos of the halo', f'{len(subhalo_ids)} loaded (those that pass the YAML subhalo filters)')
    dm_count_ok = len(own.dm_m) == totals['dm_len']
    rep.add('OK' if dm_count_ok else 'FAIL', 'DM particle count equals SubhaloLenType[:, 1]',
            f"loaded {len(own.dm_m)}, catalogue {totals['dm_len']}")

    dm_ratio = ratio(own.dm_m.sum(), totals['dm_mass'] * MASS_UNIT)
    rep.add('OK' if abs(dm_ratio - 1) < DM_MASS_REL_TOL else 'FAIL', 'DM mass equals SubhaloMassType[:, 1] * 1e10',
            f'ratio {dm_ratio:.6f}')

    star_ratio = ratio(own.star_m.sum(), totals['star_mass'] * MASS_UNIT)
    removed = totals['star_len'] - len(own.star_m)
    rep.add('OK' if removed >= 0 and 0.99 <= star_ratio <= 1 + DM_MASS_REL_TOL else 'FAIL',
            'stellar mass matches SubhaloMassType[:, 4] (the wind cells are removed)',
            f'ratio {star_ratio:.6f}, {removed} of {totals["star_len"]} catalogue star cells removed '
            f'({ratio(removed, totals["star_len"]):.4%})')

    group = lookup.group(halo.halo_id)
    group_dm = float(group['GroupMassType'][1]) * MASS_UNIT
    captured = ratio(own.dm_m.sum(), group_dm)
    rep.add('OK' if captured >= 0.5 else 'WARN', 'share of the FoF group dark matter held by the loaded subhalos',
            f'{captured:.1%}. The rest is unbound or in subhalos below the YAML filters, the DM target map does not '
            f'include it')


# ---------------------------------------------------------------------------------------------- geometry

def check_geometry(rep, own, halo):
    r200 = float(halo.r200)
    m200 = float(halo.m200)

    def centre(positions, masses):
        return np.average(positions, axis=0, weights=masses)

    offsets = {}
    for label, positions, masses in (('stars', own.star_p, own.star_m), ('dark matter', own.dm_p, own.dm_m)):
        offsets[label] = np.linalg.norm(centre(positions, masses)) / r200
        median = float(np.median(np.linalg.norm(positions, axis=1))) / r200
        low, high = RADIUS_FAIL_RANGE
        rep.add('OK' if low < median < high else 'FAIL', f'{label}: median radius in units of R200 (R200={r200:.1f})',
                f'{median:.3f}, centre offset {offsets[label]:.3f} R200')
        if offsets[label] > CENTRE_OFFSET_WARN:
            rep.add('WARN', f'{label} centre is far from the halo centre', f'{offsets[label]:.2f} R200')

    separation = np.linalg.norm(centre(own.star_p, own.star_m) - centre(own.dm_p, own.dm_m)) / r200
    rep.add('OK' if separation < CENTRE_OFFSET_WARN else 'FAIL', 'star and dark matter centres are in the same place',
            f'{separation:.3f} R200 apart')

    inside = float(own.dm_m[np.linalg.norm(own.dm_p, axis=1) < r200].sum())
    fraction = ratio(inside, m200)
    low, high = DM_FRACTION_OF_M200_WARN
    rep.add('OK' if low <= fraction <= high else 'WARN', 'DM mass inside R200 over M200 (about 0.84 for a complete halo)',
            f'{fraction:.3f} (M200={m200:.4g} Msun/h)')


# ---------------------------------------------------------------------------------------------- maps

def check_maps(rep, cfg, own, halo, features):
    catalog = cfg.catalog
    radius = float(catalog.transition_radius)
    bins = muse_bins_r(radius * 2, catalog.spatial_resolution, catalog.bin_spatial_resolution_scale)
    ranging = ((-radius, radius), (-radius, radius))
    rep.add('INFO', 'map grid', f'{bins}x{bins} bins over +-{radius} kpc, pixel {2 * radius / bins:.3f} kpc, '
                                f'zoom {2 * int(bins / 4)}x{2 * int(bins / 4)}')
    input_feature, output_feature = features
    for projection, axis in PROJECTIONS.items():
        z_map = input_feature.generate_map(cfg, halo, own, bins, ranging, axis=axis)[0]
        m_map = output_feature.generate_map(cfg, halo, own, bins, ranging, axis=axis)[0]
        i, j = axis
        ok = z_map.shape == m_map.shape == (bins, bins) and np.isfinite(z_map).all() and np.isfinite(m_map).all()
        ok = ok and (z_map >= 0).all() and (m_map >= 0).all()
        rep.add('OK' if ok else 'FAIL', f'{projection}: maps have shape ({bins}, {bins}), are finite and non-negative')

        expected = np.histogram2d(own.dm_p[:, i], own.dm_p[:, j], bins=bins, range=ranging, weights=own.dm_m)[0]
        rep.add('OK' if np.allclose(m_map, expected, rtol=1e-9, atol=0) else 'FAIL',
                f'{projection}: DM map equals an independent histogram of dm_m on dm_p')
        in_window = ratio(m_map.sum(), own.dm_m.sum())
        rep.add('INFO', f'{projection}: DM map', f'{in_window:.1%} of the halo DM mass inside the window, '
                                                  f'{(m_map > 0).mean():.1%} non-empty pixels, max pixel {m_map.max():.4g} Msun/h')
        filled = z_map[z_map > 0]
        rep.add('OK' if filled.size and filled.max() <= own.star_metallicity.max() + 1e-12 else 'FAIL',
                f'{projection}: metallicity map stays within the particle values',
                f'{(z_map > 0).mean():.1%} non-empty pixels, range {filled.min():.4g} to {filled.max():.4g}'
                if filled.size else 'no non-empty pixel')


# ---------------------------------------------------------------------------------------------- neighbour

def check_neighbour(rep, own, halo_context, halo_id, halo_ids, fetcher):
    position = halo_ids.index(halo_id)
    if position + 1 >= len(halo_ids):
        rep.add('INFO', 'neighbour check skipped', 'no next halo in the list')
        return
    other_id = halo_ids[position + 1]
    other = halo_context.get_halo_by_id(other_id)
    subhalo_id = int(list(other.subhalo_ids)[0])
    neighbour = fetcher.get_p_subhalo(subhalo_id=subhalo_id, halo_id=other_id)
    rep.add('OK' if neighbour.particle_types == ['star'] else 'FAIL', 'a neighbour subhalo loads stars only',
            f'subhalo {subhalo_id} of halo {other_id}: types {neighbour.particle_types}')

    own_stars, own_dm = len(own.star_m), len(own.dm_m)
    combined = own + neighbour
    recovered = combined.non_overlapping
    ok = len(recovered.star_m) == own_stars and len(recovered.dm_m) == own_dm
    ok = ok and len(combined.star_m) == own_stars + len(neighbour.star_m) and len(combined.dm_m) == own_dm
    rep.add('OK' if ok else 'FAIL', "the overlap boundaries recover the halo's own particles of every type",
            f'boundaries {combined.overlapping_boundaries}')


def check_rotation_guard(rep, own):
    positions = own.star_p.copy(), own.dm_p.copy()
    try:
        own.rotate(np.array([0, 0, 1]), 45)
    except NotImplementedError:
        unchanged = np.array_equal(own.star_p, positions[0]) and np.array_equal(own.dm_p, positions[1])
        rep.add('OK' if unchanged else 'FAIL', 'rotation is refused for several particle types and changes nothing')
        return
    rep.add('FAIL', 'rotation with several particle types was accepted', 'star_p and dm_p could be misaligned')


# ---------------------------------------------------------------------------------------------- one halo

def check_halo(rep, cfg, fetcher, halo_context, halo_id, halo_ids, lookup, mass_table, features):
    halo = halo_context.get_halo_by_id(halo_id)
    print(f'\n  -- halo {halo_id}: m200={float(halo.m200):.4g}, r200={float(halo.r200):.1f}, '
          f'{len(halo.subhalo_ids)} subhalos')
    before = current_rss_mb()
    start = time.perf_counter()
    own = fetcher.get_p_halo(halo=halo)
    elapsed = time.perf_counter() - start
    millions = (len(own.star_m) + len(own.dm_m)) / 1e6 if hasattr(own, 'star_m') and hasattr(own, 'dm_m') else float('nan')
    rep.add('INFO', 'load time', f'{elapsed:.1f} s for {millions:.1f} million particles ({ratio(elapsed, millions):.1f} s per million)')
    rep.add('INFO', 'memory', f'{current_rss_mb():.0f} MB now ({current_rss_mb() - before:+.0f} MB for this halo), peak so far {peak_rss_mb():.0f} MB')

    sample = (float(halo.m200), millions, elapsed)
    if not check_particles(rep, own, mass_table):
        return sample
    guarded(rep, 'catalogue check', check_catalogue, rep, own, halo, lookup)
    own.preprocess(halo, cfg.catalog.get('Lbox', None))
    guarded(rep, 'geometry check', check_geometry, rep, own, halo)
    for feature in features:
        feature.process(halo=halo, subhalo_particles=own)
    start = time.perf_counter()
    guarded(rep, 'map check', check_maps, rep, cfg, own, halo, features)
    rep.add('INFO', 'map generation time (3 projections, 2 features)', f'{time.perf_counter() - start:.1f} s')
    guarded(rep, 'neighbour check', check_neighbour, rep, own, halo_context, halo_id, halo_ids, fetcher)
    guarded(rep, 'rotation check', check_rotation_guard, rep, own)
    return sample


# ---------------------------------------------------------------------------------------------- planning

def estimate_full_run(rep, m200_values, dm_particle_mass, samples):
    """
    Rough planning estimate of the particle loading time for EVERY selected halo.

    `samples` are the halos checked above, as (m200, millions of particles loaded, seconds to load them). The
    particles loaded per halo are of the order of M200 / m_dm, with a factor that depends on the halo (the FoF group
    reaches beyond R200), measured here as particles / (M200 / m_dm). The estimate takes the lowest, the median and
    the highest of the measured factors and of the measured seconds per million particles. It covers the particle
    loading only, not the catalogue listing, the neighbours (stars only) nor the map generation.
    """
    samples = [sample for sample in samples if sample]
    if len(samples) < 2 or dm_particle_mass <= 0:
        rep.add('INFO', 'planning estimate skipped', 'needs at least two loaded halos')
        return
    factors = np.array([millions * 1e6 / (m200 / dm_particle_mass) for m200, millions, _ in samples])
    speeds = np.array([seconds / millions for _, millions, seconds in samples])
    expected = np.asarray(m200_values, dtype=float) / dm_particle_mass          # DM particles inside M200, per halo
    millions = {name: expected.sum() * factor / 1e6 for name, factor in
                (('low', factors.min()), ('median', np.median(factors)), ('high', factors.max()))}
    hours = {'low': millions['low'] * speeds.min() / 3600, 'median': millions['median'] * np.median(speeds) / 3600,
             'high': millions['high'] * speeds.max() / 3600}
    rep.add('INFO', 'measured', f'{factors.min():.2f} to {factors.max():.2f} particles per M200/m_dm, '
                                f'{speeds.min():.1f} to {speeds.max():.1f} s per million particles')
    rep.add('INFO', f'planning estimate for loading the particles of all {len(expected)} selected halos',
            f"{millions['median']:.0f} million particles in total (range {millions['low']:.0f} to {millions['high']:.0f}), "
            f"about {hours['median']:.1f} h in one process (range {hours['low']:.1f} to {hours['high']:.1f} h)")


# ---------------------------------------------------------------------------------------------- output

def check_output_directory(rep, cfg):
    root = os.path.join(cfg.catalog.path, f'no_rotation_s{cfg.catalog.bin_spatial_resolution_scale}')
    labels = [feature.get('label') or feature.attribute for feature in cfg.feature_names]
    if not os.path.isdir(root):
        rep.add('INFO', 'generated maps', f'none yet, {root} does not exist (run the pipeline on a few halos)')
        return
    folders = {label: os.path.join(root, label) for label in labels}
    missing = [label for label, folder in folders.items() if not os.path.isdir(folder)]
    if missing:
        rep.add('FAIL', 'feature folders', f'missing {missing} under {root}')
        return
    names = {label: sorted(os.listdir(folder)) for label, folder in folders.items()}
    rep.add('INFO', 'generated maps', ', '.join(f'{label}: {len(files)} files' for label, files in names.items()))
    same = names[labels[0]] == names[labels[1]]
    rep.add('OK' if same else 'FAIL', 'input and output folders hold the same file names')
    bad_names = [n for n in names[labels[0]] if not FILE_PATTERN.fullmatch(n)]
    rep.add('OK' if not bad_names else 'FAIL', 'file names follow halo_<id>_subhalo_<id>_<projection>_zoom_<i>.npz',
            str(bad_names[:3]) if bad_names else '')

    parsed = [FILE_PATTERN.fullmatch(n) for n in names[labels[0]] if FILE_PATTERN.fullmatch(n)]
    halos = sorted({int(m.group(1)) for m in parsed})
    rep.add('INFO', 'halos with maps', f'{len(halos)}: {halos[:10]}')
    bins = muse_bins_r(float(cfg.catalog.transition_radius) * 2, cfg.catalog.spatial_resolution,
                       cfg.catalog.bin_spatial_resolution_scale)
    side = 2 * int(bins / 4)

    # Only files present in both folders can be compared, a difference was already reported above
    sample = sorted(set(names[labels[0]]) & set(names[labels[1]]))[:MAX_OUTPUT_FILES_TO_READ]
    stats = {'masked': 0, 'mask_pixels': 0, 'z_max': 0.0, 'dm_min': np.inf, 'bad_shape': 0, 'nonfinite': 0}
    for name in sample:
        with np.load(os.path.join(folders[labels[0]], name)) as z_file, np.load(os.path.join(folders[labels[1]], name)) as m_file:
            z, m = z_file['arr_0'], m_file['arr_0']
        stats['bad_shape'] += int(z.shape != (side, side, 2) or m.shape != (side, side, 2))
        stats['nonfinite'] += int(not (np.isfinite(z).all() and np.isfinite(m).all()))
        stats['z_max'] = max(stats['z_max'], float(z[..., 0].max()))
        stats['dm_min'] = min(stats['dm_min'], float(m[..., 0].min()))
        stats['masked'] += int(z[..., 1].any())
        stats['mask_pixels'] += int(z[..., 1].sum())
        if not np.array_equal(z[..., 1], m[..., 1]):
            rep.add('FAIL', f'{name}: input and output files have different masks')
    rep.add('OK' if not stats['bad_shape'] else 'FAIL', f'each file is an array of shape ({side}, {side}, 2)',
            f"{stats['bad_shape']} of {len(sample)} wrong" if stats['bad_shape'] else '')
    rep.add('OK' if not stats['nonfinite'] else 'FAIL', 'no NaN or inf in the sampled files')
    rep.add('OK' if stats['z_max'] <= 1 and stats['dm_min'] >= 0 else 'FAIL', 'value ranges',
            f"max metallicity {stats['z_max']:.4g}, min DM {stats['dm_min']:.4g}")
    # The mask channel marks the interloper subhalos. If it is empty everywhere, interlopers are not masked
    rep.add('OK' if stats['masked'] else 'WARN', 'interloper mask channel',
            f"{stats['masked']} of {len(sample)} sampled files have masked pixels ({stats['mask_pixels']} in total)"
            + ('' if stats['masked'] else '. All zero: check MapStep._get_interlopers_mask, which only looks up '
                                          'subhalos whose id is a python int'))


# ---------------------------------------------------------------------------------------------- main

def check_catalog(catalog_name):
    print(f'\n=== {catalog_name} ===')
    rep = Report(catalog_name)
    cfg = load_config(catalog_name)
    fetcher = DataLoader(cfg).fetcher
    if not check_dataset_directory(rep, cfg, fetcher):
        return rep
    lookup = CatalogLookup(os.path.join(cfg.catalog.path, 'output'), cfg.catalog.snapnum)

    start = time.perf_counter()
    halo_context = fetcher.get_halos()
    halo_ids = [int(i) for i in halo_context.halo_id]
    rep.add('INFO', 'halos after the catalog filters', f'{len(halo_ids)}, listed in {time.perf_counter() - start:.0f} s')
    rep.add('INFO', 'memory after listing the halos (catalogue only, no particles)', f'{current_rss_mb():.0f} MB now, peak {peak_rss_mb():.0f} MB')
    positions = [p for p in HALO_POSITIONS if p < len(halo_ids)]
    if len(positions) < len(HALO_POSITIONS):
        rep.add('WARN', 'halo positions beyond the list were ignored', str([p for p in HALO_POSITIONS if p >= len(halo_ids)]))

    features = [FeatureProcessorFactory.get_feature(feature, cfg=cfg) for feature in cfg.feature_names]
    mass_table = fetcher._get_mass_table()
    samples = []
    try:
        for position in positions:
            try:
                samples.append(check_halo(rep, cfg, fetcher, halo_context, halo_ids[position], halo_ids, lookup,
                                          mass_table, features))
            except Exception as error:      # noqa: BLE001
                rep.add('FAIL', f'halo {halo_ids[position]} raised an exception', f'{type(error).__name__}: {error}')
    finally:
        lookup.close()
    print('\n  -- planning')
    estimate_full_run(rep, halo_context.data['m200'], mass_table[1] * MASS_UNIT, samples)
    print('\n  -- generated maps')
    check_output_directory(rep, cfg)
    return rep


def main():
    reports = []
    for catalog in CATALOGS:
        try:
            reports.append(check_catalog(catalog))
        except Exception as error:      # noqa: BLE001
            rep = Report(catalog)
            rep.add('FAIL', 'could not run the checks', f'{type(error).__name__}: {error}')
            reports.append(rep)

    print('\n=== summary ===')
    for rep in reports:
        counts = {s: sum(1 for line in rep.lines if line['status'] == s) for s in ('OK', 'WARN', 'FAIL')}
        print(f'  {rep.name:<18} OK={counts["OK"]} WARN={counts["WARN"]} FAIL={counts["FAIL"]}')
    print(f'  peak memory of the whole run: {peak_rss_mb():.0f} MB')
    if JSON_OUTPUT:
        with open(JSON_OUTPUT, 'w') as f:
            json.dump({rep.name: rep.lines for rep in reports}, f, indent=2)
        print(f'report written to {JSON_OUTPUT}')
    sys.exit(1 if any(rep.failed for rep in reports) else 0)


if __name__ == '__main__':
    main()
