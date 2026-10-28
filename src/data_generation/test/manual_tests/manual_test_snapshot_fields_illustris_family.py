"""
Standalone field check for the Illustris-family catalogs (illustris300, tngcluster, millenium).

Not a pytest fixture, run it directly on a machine that can see the snapshots
(MN5). It only needs h5py and numpy, not the pipeline or illustris_python.

Purpose: before adding stellar metallicity and dark-matter mass maps, confirm
that each simulation's snapshot actually contains the variables those features
need, and that their units behave as assumed. For each simulation it finds the
first chunk of the snapshot and of the group catalogue, and checks:

  Header      scale factor vs the expected `a`, BoxSize vs `Lbox`, MassTable,
              NumPart_Total (incl. the high word), number of files
  PartType4   Coordinates, Masses, Velocities, GFM_Metallicity and
              GFM_StellarFormationTime are present, plus a sample of
              metallicity statistics and the wind-particle fraction
  PartType1   Coordinates present, and whether the DM mass comes from a
              Masses dataset or only from the header MassTable (and whether
              MassTable[1] is close to the published DM particle mass)
  Groups      SubhaloMassType, SubhaloLenType and the subhalo parent field
              (SubhaloGrNr or SubhaloGroupNr). Cross-checks
              SubhaloMassType[:, 1] against SubhaloLenType[:, 1] * MassTable[1],
              which validates the DM mass unit end to end.

Only the FIRST chunk file of each snapshot is read (and at most MAX_ROWS rows
of each dataset), so the statistics are a sample, not the whole box. Chunk 0
is not guaranteed to be representative of the full volume.

Parameters are hardcoded in the SIMULATIONS table below (copied from
configs/catalog/{illustris300,tngcluster,millenium}.yaml). Edit DATASETS_DIR or
an entry if your data lives elsewhere, then run directly:
    python src/data_generation/test/manual_tests/manual_test_snapshot_fields_illustris_family.py

Only the FIRST chunk file of each snapshot is read (and at most MAX_ROWS rows
of each dataset), so the statistics are a sample, not the whole box. Chunk 0
is not guaranteed to be representative of the full volume.

Usage:
    # 1) parameters passed directly, no YAML needed (one simulation per call)
    python ... --path /path/to --snapnum 84 --a 0.8333 --lbox 680000 \
               --expected-masstable 0.0041 --name tngcluster
    # 2) read the parameters from the catalog YAMLs (all three, or a subset)
    python ...
    python ... --datasets-dir /path/to/datasets --only tngcluster --json report.json

In direct mode --path is the simulation folder, or the `output` folder itself:
the script looks for snapdir_NNN/ and groups_NNN/ in either. --a, --lbox and
--expected-masstable are optional, and the matching checks are skipped when
they are not given.

Exit code is 1 if any FAIL is reported, so it can gate a job.
"""
import glob
import json
import os
import re
import sys
import time

import h5py
import numpy as np

DATASETS_DIR = '/path/to'

# path: simulation folder (or its `output` folder), containing snapdir_NNN/ and groups_NNN/
# a, Lbox: from the catalog YAMLs, checked against the snapshot header (None to skip)
# expected_masstable: approximate MassTable[1] in header units (1e10 Msun/h), derived from
# the published DM particle masses assuming h = 0.6774 (TNG300 ~5.9e7 Msun, TNG-Cluster
# ~6.1e7 Msun, MTNG-L500-4320-A 1.12e8 Msun/h). Only a sanity range, not a documented value.
SIMULATIONS = {
    'illustris300': {'path': f'{DATASETS_DIR}/illustris300', 'snapnum': 84, 'a': 0.8333,
                     'Lbox': 205000, 'expected_masstable': 0.0040},
    'tngcluster': {'path': f'{DATASETS_DIR}/tng_cluster', 'snapnum': 84, 'a': 0.8333,
                   'Lbox': 680000, 'expected_masstable': 0.0041},
    'millenium': {'path': f'{DATASETS_DIR}/millenium/MTNG-L500-4320-A', 'snapnum': 237, 'a': 0.8,
                  'Lbox': 500000, 'expected_masstable': 0.0112},
}
ONLY = ['tngcluster']          # e.g. ['tngcluster'] to check a subset
# Simulations whose group catalogue is scanned in FULL (every chunk, only 3 small datasets each) to count
# how many massive halos are free of low-resolution (type-2) contamination. Only useful for zoom
# simulations, TNG300 and MTNG have no type-2 particles. Takes a few minutes on GPFS.
SCAN_ALL_CHUNKS_FOR = ['tngcluster']
CONTAMINATION_THRESHOLDS = [0.0, 0.001, 0.01, 0.1]   # type-2 mass fraction cuts to report
# Halo mass cut of the catalogs: filter "> 1e13 * 0.6774" on m200 in Msun/h, i.e. 677.4 in header
# units (1e10 Msun/h). The scans use the same cut so their counts match what the pipeline selects.
MIN_M200 = 1e13 * 0.6774 / 1e10
JSON_OUTPUT = None   # e.g. 'report.json' to also save the full report

EXPECTED_TOLERANCE = 0.25  # relative

STAR_FIELDS = ['Coordinates', 'Masses', 'Velocities', 'GFM_Metallicity', 'GFM_StellarFormationTime']
MAX_ROWS = 2_000_000


class Report:
    """Collects (status, label, detail) lines for one catalog and prints them as it goes."""

    def __init__(self, name):
        self.name = name
        self.lines = []

    def add(self, status, label, detail=''):
        self.lines.append({'status': status, 'label': label, 'detail': detail})
        print(f'  [{status:<4}] {label}' + (f': {detail}' if detail else ''))

    @property
    def failed(self):
        return any(line['status'] == 'FAIL' for line in self.lines)


def resolve_output_dir(path):
    """Accept either the simulation folder or its `output` subfolder."""
    candidate = os.path.join(path, 'output')
    return candidate if os.path.isdir(candidate) else path


def first_chunk(directory, pattern):
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    return matches[0] if matches else None


def find_snapshot_chunk(snapdir, snap, rep):
    """
    Chunk 0 of the FULL snapshot. Prefers snap_NNN.0 / snapshot_NNN.0 and never picks reduced
    files such as 'snapshot-prevmostboundonly_NNN.0' (MTNG), which hold only a subset of particles.
    """
    if not os.path.isdir(snapdir):
        return None
    candidates = sorted(glob.glob(os.path.join(snapdir, '*.0.hdf5')))
    rep.add('INFO', 'chunk-0 files in snapdir', str([os.path.basename(c) for c in candidates]))
    for name in (f'snap_{snap:03d}.0.hdf5', f'snapshot_{snap:03d}.0.hdf5'):
        path = os.path.join(snapdir, name)
        if os.path.exists(path):
            return path
    others = [c for c in candidates if 'prevmostbound' not in os.path.basename(c)]
    return others[0] if others else None


def header_summary(h5, rep, cat):
    hdr = dict(h5['Header'].attrs)
    masstable = np.asarray(hdr.get('MassTable', np.zeros(6)), dtype=float)
    low = np.asarray(hdr.get('NumPart_Total', np.zeros(6)), dtype=np.uint64)
    high = np.asarray(hdr.get('NumPart_Total_HighWord', np.zeros(6)), dtype=np.uint64)
    n_total = (high << np.uint64(32)) + low

    rep.add('INFO', 'Header',
            f"Time={hdr.get('Time')}, Redshift={hdr.get('Redshift')}, h={hdr.get('HubbleParam')}, "
            f"BoxSize={hdr.get('BoxSize')}, files={hdr.get('NumFilesPerSnapshot')}")
    rep.add('INFO', 'NumPart_Total (gas, dm, type2, tracers, stars, bh)', str([int(x) for x in n_total]))
    rep.add('INFO', 'MassTable', str(masstable.tolist()))

    time = hdr.get('Time')
    if cat['a'] is not None and time is not None:
        rel = abs(float(time) - float(cat['a'])) / float(cat['a'])
        rep.add('OK' if rel < 1e-4 else 'WARN', 'scale factor matches expected a',
                f"header a={float(time):.5f}, expected a={cat['a']} (rel diff {rel:.2%})")
    if cat['Lbox'] is not None and hdr.get('BoxSize') is not None:
        box = float(hdr['BoxSize'])
        lbox = float(cat['Lbox'])
        if abs(box - lbox) / lbox < 1e-3:
            rep.add('OK', 'BoxSize matches expected Lbox', f'{box} (kpc/h)')
        elif abs(box * 1000 - lbox) / lbox < 1e-3:
            rep.add('OK', 'BoxSize matches expected Lbox', f'{box} is in Mpc/h, x1000 = {box * 1000:g} kpc/h')
        else:
            rep.add('WARN', 'BoxSize matches expected Lbox', f'header={box}, expected={lbox}')
    return masstable, n_total


def check_stars(h5, rep):
    if 'PartType4' not in h5:
        rep.add('FAIL', 'PartType4 group', 'missing in this chunk')
        return
    star = h5['PartType4']
    for field in STAR_FIELDS:
        if field in star:
            ds = star[field]
            rep.add('OK', f'PartType4/{field}', f'shape={ds.shape}, dtype={ds.dtype}')
        else:
            level = 'FAIL' if field == 'GFM_Metallicity' else 'WARN'
            rep.add(level, f'PartType4/{field}', 'MISSING')
    if 'GFM_StellarFormationTime' in star:
        t = star['GFM_StellarFormationTime'][:MAX_ROWS]
        wind = float(np.mean(t <= 0))
        rep.add('INFO', 'wind fraction in PartType4 sample (formation time <= 0)', f'{wind:.2%}')
        if wind > 0:
            rep.add('WARN', 'wind particles are present',
                    'a filter on GFM_StellarFormationTime > 0 is needed to keep real stars only')

    if 'GFM_Metallicity' not in star:
        candidates = [k for k in star.keys() if 'etal' in k]
        rep.add('INFO', 'metallicity-like PartType4 fields', str(candidates))
        return

    z = star['GFM_Metallicity'][:MAX_ROWS]
    if z.ndim > 1:
        rep.add('WARN', 'GFM_Metallicity is not 1-D', f'shape={z.shape}')
        return
    n_nan = int(np.isnan(z).sum())
    rep.add('INFO', 'GFM_Metallicity sample', f'n={len(z)}, min={np.nanmin(z):.4g}, mean={np.nanmean(z):.4g}, '
                                              f'max={np.nanmax(z):.4g}, NaN={n_nan}')
    # total metal mass fraction, so it must live in [0, 1] and be of order 1e-2
    in_range = bool(np.nanmin(z) >= 0 and np.nanmax(z) <= 1)
    rep.add('OK' if in_range else 'WARN', 'metallicity looks like a mass fraction in [0, 1]')


def check_dm(h5, rep, masstable, expected):
    if 'PartType1' not in h5:
        rep.add('FAIL', 'PartType1 group', 'missing in this chunk')
        return
    dm = h5['PartType1']
    rep.add('INFO', 'PartType1 fields', str(sorted(dm.keys())))
    rep.add('OK' if 'Coordinates' in dm else 'FAIL', 'PartType1/Coordinates')

    if 'Masses' in dm:
        m = dm['Masses'][:MAX_ROWS]
        n_unique = len(np.unique(m))
        rep.add('INFO', 'PartType1/Masses present', f'{n_unique} distinct value(s) in sample, min={m.min():.4g}, max={m.max():.4g}')
        if n_unique > 1:
            rep.add('WARN', 'DM masses are NOT uniform', 'a header-only mass would be wrong, read Masses per particle')
    else:
        rep.add('INFO', 'PartType1 has no Masses dataset', 'DM mass must come from the header MassTable')
        if masstable[1] > 0:
            rep.add('OK', 'MassTable[1] is non-zero', f'{masstable[1]:.5g} (1e10 Msun/h)')
        else:
            rep.add('FAIL', 'MassTable[1] is zero and PartType1 has no Masses', 'no way to recover the DM mass')

    if expected is not None and masstable[1] > 0:
        rel = abs(masstable[1] - expected) / expected
        rep.add('OK' if rel < EXPECTED_TOLERANCE else 'WARN', 'MassTable[1] vs published DM mass (approx)',
                f'header={masstable[1]:.5g}, expected about {expected} (rel diff {rel:.0%})')
    if masstable[2] > 0:
        rep.add('INFO', 'MassTable[2] is non-zero', f'{masstable[2]:.5g}, type-2 (low-resolution) DM may exist')
    if 'PartType2' in h5:
        t2 = h5['PartType2']
        rep.add('INFO', 'PartType2 fields (low-resolution DM in zooms)', str(sorted(t2.keys())))
        if 'Masses' in t2:
            m2 = t2['Masses'][:MAX_ROWS]
            rep.add('INFO', 'PartType2/Masses sample', f'distinct values={len(np.unique(m2))}, min={m2.min():.4g}, max={m2.max():.4g}')
        elif masstable[2] == 0:
            rep.add('WARN', 'PartType2 exists but has no Masses and MassTable[2] is zero', 'its particle mass is not recoverable')


def check_groups(group_file, rep, masstable):
    with h5py.File(group_file, 'r') as g:
        if 'Subhalo' not in g:
            rep.add('FAIL', 'Subhalo group', f'missing in {os.path.basename(group_file)}')
            return
        sub = g['Subhalo']
        for field in ['SubhaloMassType', 'SubhaloLenType']:
            rep.add('OK' if field in sub else 'FAIL', f'Subhalo/{field}', f'shape={sub[field].shape}' if field in sub else 'MISSING')
        parent = [f for f in ('SubhaloGrNr', 'SubhaloGroupNr') if f in sub]
        rep.add('OK' if parent else 'FAIL', 'subhalo parent field', str(parent) if parent else 'neither SubhaloGrNr nor SubhaloGroupNr found')

        if 'SubhaloMassType' in sub and 'SubhaloLenType' in sub and masstable[1] > 0:
            mass_dm = sub['SubhaloMassType'][:MAX_ROWS, 1]
            len_dm = sub['SubhaloLenType'][:MAX_ROWS, 1]
            sel = len_dm > 0
            if sel.any():
                ratio = mass_dm[sel] / (len_dm[sel] * masstable[1])
                ok = abs(np.median(ratio) - 1) < 0.01
                rep.add('OK' if ok else 'WARN', 'SubhaloMassType[:,1] == SubhaloLenType[:,1] * MassTable[1]',
                        f'ratio median={np.median(ratio):.4f}, min={ratio.min():.4f}, max={ratio.max():.4f} over {int(sel.sum())} subhalos')
            else:
                rep.add('INFO', 'DM mass cross-check skipped', 'no subhalo with DM in this group chunk')

        if 'SubhaloLenType' in sub and 'SubhaloMassType' in sub:
            len_all = sub['SubhaloLenType'][:MAX_ROWS]
            mass_all = sub['SubhaloMassType'][:MAX_ROWS]
            n_with_t2 = int((len_all[:, 2] > 0).sum())
            if n_with_t2 > 0:
                frac = float(mass_all[:, 2].sum() / max(mass_all[:, 1].sum(), 1e-30))
                rep.add('WARN', 'subhalos contain type-2 particles',
                        f'{n_with_t2} subhalos, type-2 mass = {frac:.2%} of type-1 mass in this chunk. '
                        'A PartType1-only DM map would miss it')
            else:
                rep.add('OK', 'no subhalo contains type-2 particles in this group chunk')


def check_contamination(group_file, rep, min_m200=None):
    """
    How much low-resolution (type-2) mass do the MASSIVE FoF groups contain?

    min_m200 is in header units (1e10 Msun/h) and defaults to MIN_M200, the catalogs' halo mass cut. This is group level (includes unbound material), a conservative measure for the
    clusters that actually get mapped.
    """
    min_m200 = MIN_M200 if min_m200 is None else min_m200
    with h5py.File(group_file, 'r') as g:
        if 'Group' not in g or not all(k in g['Group'] for k in ('GroupMassType', 'Group_M_Crit200')):
            rep.add('INFO', 'cluster contamination check skipped', 'Group/GroupMassType or Group_M_Crit200 missing')
            return
        grp = g['Group']
        m200 = grp['Group_M_Crit200'][:MAX_ROWS]
        mass = grp['GroupMassType'][:MAX_ROWS]
        # TNG-Cluster flags its 352 zoom targets (believed to be GroupPrimaryZoomTarget, to be confirmed by this run)
        flag_name = next((k for k in grp.keys() if 'zoomtarget' in k.lower()), None)
        target = grp[flag_name][:MAX_ROWS] if flag_name else None
    rep.add('INFO', 'zoom-target flag in Group', flag_name or 'none found')

    def summarise(label, sel):
        n = int(sel.sum())
        if n == 0:
            rep.add('INFO', label, 'no groups')
            return
        dm_hi, dm_lo = mass[sel, 1], mass[sel, 2]
        frac = dm_lo / np.maximum(dm_hi + dm_lo, 1e-30)
        c = [int((frac <= 0).sum()), int(((frac > 0) & (frac <= 0.01)).sum()), int(((frac > 0.01) & (frac <= 0.1)).sum()),
             int(((frac > 0.1) & (frac <= 0.9)).sum()), int((frac > 0.9).sum())]
        detail = (f'{n} groups, contamination median={np.median(frac):.3%}, p90={np.percentile(frac, 90):.3%}, max={frac.max():.3%}. '
                  f'Counts: =0%: {c[0]}, (0,1%]: {c[1]}, (1,10%]: {c[2]}, (10,90%]: {c[3]}, >90%: {c[4]}')
        rep.add('WARN' if (dm_lo > 0).any() else 'OK', label, detail)

    sel = m200 >= min_m200
    summarise(f'type-2 contamination, groups with M200 >= {min_m200:g} (1e10 Msun/h)', sel)
    if target is not None:
        summarise('type-2 contamination, zoom targets only', target.astype(bool))
        summarise('type-2 contamination, massive groups that are NOT zoom targets', sel & ~target.astype(bool))


def scan_all_group_chunks(groupdir, snap, rep, min_m200=None):
    """
    Whole-box version of check_contamination: reads Group_M_Crit200, GroupMassType and (if present) the
    zoom-target flag from EVERY group-catalogue chunk, then reports how many massive groups stay below
    each contamination threshold, separately for zoom targets and for the other massive groups.
    """
    def chunk_no(path):
        return int(re.search(r'\.(\d+)\.hdf5$', path).group(1))

    min_m200 = MIN_M200 if min_m200 is None else min_m200
    files = sorted(glob.glob(os.path.join(groupdir, f'fof_subhalo_tab_{snap:03d}.*.hdf5')), key=chunk_no)
    if not files:
        rep.add('FAIL', 'full group scan', f'no fof_subhalo_tab files under {groupdir}')
        return
    start = time.time()
    n_groups, n_empty = 0, 0
    m200_l, frac_l, tgt_l, file_l = [], [], [], []
    flag_name = None
    for i, path in enumerate(files):
        with h5py.File(path, 'r') as g:
            if 'Group' not in g or 'Group_M_Crit200' not in g['Group']:
                n_empty += 1
                continue
            grp = g['Group']
            m200 = grp['Group_M_Crit200'][:]
            mass = grp['GroupMassType'][:]
            if flag_name is None:
                flag_name = next((k for k in grp.keys() if 'zoomtarget' in k.lower()), None)
            tgt = grp[flag_name][:].astype(bool) if flag_name else np.zeros(len(m200), dtype=bool)
        n_groups += len(m200)
        keep = (m200 >= min_m200) | tgt
        total = mass[keep, 1] + mass[keep, 2]
        m200_l.append(m200[keep])
        frac_l.append(mass[keep, 2] / np.maximum(total, 1e-30))
        tgt_l.append(tgt[keep])
        file_l.append(np.full(int(keep.sum()), chunk_no(path)))
        if (i + 1) % 100 == 0:
            print(f'    ... {i + 1}/{len(files)} group chunks scanned ({time.time() - start:.0f}s)')

    if not m200_l:
        rep.add('FAIL', 'full group scan', 'no usable group data in any chunk')
        return
    m200, frac, tgt, chunk = (np.concatenate(x) for x in (m200_l, frac_l, tgt_l, file_l))
    massive = m200 >= min_m200
    rep.add('INFO', 'full group scan', f'{len(files)} chunks ({n_empty} without groups) in {time.time() - start:.0f}s, '
                                       f'{n_groups} groups, {int(massive.sum())} massive (M200 >= {min_m200:g}), '
                                       f'zoom-target flag: {flag_name or "none"}, {int(tgt.sum())} targets')

    def counts(label, sel):
        n = int(sel.sum())
        if n == 0:
            rep.add('INFO', label, 'no groups')
            return
        parts = ', '.join(f'<= {t:.1%}: {int((frac[sel] <= t).sum())}' for t in CONTAMINATION_THRESHOLDS)
        rep.add('INFO', label, f'{n} groups. Clean counts ({"type-2 mass fraction"}): {parts}')

    counts('massive groups, all', massive)
    if flag_name:
        counts('zoom targets', tgt)
        counts('massive groups that are not zoom targets', massive & ~tgt)
        bad = tgt & (frac > 0.01)
        if bad.any():
            worst = np.argsort(-frac[bad])[:10]
            listing = ', '.join(f'chunk {int(chunk[bad][k])}: {frac[bad][k]:.1%}' for k in worst)
            rep.add('WARN', 'zoom targets with contamination above 1%', f'{int(bad.sum())} of {int(tgt.sum())}. Worst: {listing}')
        else:
            rep.add('OK', 'all zoom targets are below 1% contamination', f'{int(tgt.sum())} targets checked')


def check_catalog(name, cat):
    print(f'\n=== {name} ===')
    rep = Report(name)
    snap = cat['snapnum']
    out = resolve_output_dir(cat['path'])
    rep.add('INFO', 'catalog path', f"{cat['path']} (snapnum {snap})")

    snapdir = os.path.join(out, f'snapdir_{snap:03d}')
    groupdir = os.path.join(out, f'groups_{snap:03d}')
    snap_file = find_snapshot_chunk(snapdir, snap, rep)
    group_file = first_chunk(groupdir, 'fof_subhalo_tab_*.0.hdf5') if os.path.isdir(groupdir) else None
    rep.add('OK' if snap_file else 'FAIL', 'full snapshot chunk 0',
            snap_file or f'no full-snapshot file under {snapdir} (only reduced files, or nothing)')
    rep.add('OK' if group_file else 'FAIL', 'group catalogue chunk 0', group_file or f'nothing under {groupdir}')

    masstable = np.zeros(6)
    if snap_file:
        with h5py.File(snap_file, 'r') as h5:
            masstable, _ = header_summary(h5, rep, cat)
            print('  -- stars')
            check_stars(h5, rep)
            print('  -- dark matter')
            check_dm(h5, rep, masstable, cat.get('expected_masstable'))
    if group_file:
        print('  -- group catalogue')
        check_groups(group_file, rep, masstable)
        check_contamination(group_file, rep)
        if name in SCAN_ALL_CHUNKS_FOR:
            print('  -- full group-catalogue scan')
            scan_all_group_chunks(groupdir, snap, rep)
    return rep


def main():
    reports = [check_catalog(name, cat) for name, cat in SIMULATIONS.items() if ONLY is None or name in ONLY]

    print('\n=== summary ===')
    for rep in reports:
        counts = {s: sum(1 for line in rep.lines if line['status'] == s) for s in ('OK', 'WARN', 'FAIL')}
        print(f'  {rep.name:<14} OK={counts["OK"]} WARN={counts["WARN"]} FAIL={counts["FAIL"]}')
    if JSON_OUTPUT:
        with open(JSON_OUTPUT, 'w') as f:
            json.dump({rep.name: rep.lines for rep in reports}, f, indent=2)
        print(f'report written to {JSON_OUTPUT}')
    sys.exit(1 if any(rep.failed for rep in reports) else 0)


if __name__ == '__main__':
    main()
