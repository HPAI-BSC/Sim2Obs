import g3read as g3
import numpy as np
import os


# ─── PATHS ────────────────────────────────────────────────────────────────────
GROUPS_DIR = '../data/magneticum/groups_120'
SNAP_BASE  = '../data/magneticum/snapdir_120/snap_120'

snap = g3.GadgetFile('../data/magneticum/snapdir_120/snap_120.0')
print(snap.header.__dict__)

box_size_raw = snap.header.BoxSize  # 352000.0 ckpc/h

# Convert to physical Mpc
h = snap.header.HubbleParam  # 0.704
a = snap.header.time          # 0.8247

box_size_pkpc = box_size_raw * a / h   # pkpc
box_size_pMpc = box_size_pkpc / 1000   # pMpc

print(f"BoxSize: {box_size_raw} ckpc/h")
print(f"BoxSize: {box_size_pkpc:.2f} pkpc")
print(f"BoxSize: {box_size_pMpc:.2f} pMpc")


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────
def load_halos(groups_dir, fields):
    group_base = os.path.join(groups_dir, os.path.basename(groups_dir).replace('groups', 'sub'))
    return g3.read_new(group_base, fields, 0, is_snap=False, multiple_files=True)


def load_subhalos(groups_dir, fields):
    group_base = os.path.join(groups_dir, os.path.basename(groups_dir).replace('groups', 'sub'))
    return g3.read_new(group_base, fields, 1, is_snap=False, multiple_files=True)


def get_subhalo_particles_strict(snap_base, groups_dir, subhalo_idx, ptype=4,
                                  blocks=['POS ', 'VEL ', 'MASS']):
    """Get particles strictly assigned to a subhalo by Subfind."""
    group_base = os.path.join(groups_dir, os.path.basename(groups_dir).replace('groups', 'sub'))

    soff = g3.read_new(group_base, ['SOFF'], 1, is_snap=False, multiple_files=True)['SOFF']
    slen = g3.read_new(group_base, ['SLEN'], 1, is_snap=False, multiple_files=True)['SLEN']
    pid = g3.read_new(group_base, ['PID '], 2, is_snap=False, multiple_files=True)['PID ']

    start = soff[subhalo_idx]
    end = start + slen[subhalo_idx]
    subhalo_pids = set(pid[start:end])

    snap_data = g3.read_new(snap_base, blocks + ['ID  '], ptype, multiple_files=True, do_flatten=False)
    snap_data = snap_data[ptype]

    mask = np.isin(snap_data['ID  '], list(subhalo_pids))
    return {b: snap_data[b][mask] for b in blocks}


def get_halo_particles_strict(snap_base, groups_dir, halo_idx, ptype=4,
                               blocks=['POS ', 'VEL ', 'MASS']):
    """Get particles strictly assigned to a halo by Subfind."""
    group_base = os.path.join(groups_dir, os.path.basename(groups_dir).replace('groups', 'sub'))

    goff = g3.read_new(group_base, ['GOFF'], 0, is_snap=False, multiple_files=True)['GOFF']
    glen = g3.read_new(group_base, ['GLEN'], 0, is_snap=False, multiple_files=True)['GLEN']
    pid = g3.read_new(group_base, ['PID '], 2, is_snap=False, multiple_files=True)['PID ']

    start = goff[halo_idx]
    end = start + glen[halo_idx]
    halo_pids = set(pid[start:end])

    snap_data = g3.read_new(snap_base, blocks + ['ID  '], ptype, multiple_files=True, do_flatten=False)
    snap_data = snap_data[ptype]

    mask = np.isin(snap_data['ID  '], list(halo_pids))
    return {b: snap_data[b][mask] for b in blocks}


print("=" * 60)
print("TEST 1: load_halos")
print("=" * 60)
halos = load_halos(GROUPS_DIR, ['MCRI', 'RCRI', 'FSUB', 'GPOS'])
print(f"Total halos:       {len(halos['MCRI'])}")
print(f"M_Crit200 [0]:     {halos['MCRI'][0]:.4e}  (x1e10 Msun/h)")
print(f"R_Crit200 [0]:     {halos['RCRI'][0]:.4f}  (ckpc/h)")
print(f"First subhalo [0]: {halos['FSUB'][0]}")
print(f"Position [0]:      {halos['GPOS'][0]}")

print()
print("=" * 60)
print("TEST 2: load_subhalos")
print("=" * 60)
subhalos = load_subhalos(GROUPS_DIR, ['SPOS', 'SMST', 'RHMS', 'GRNR'])
print(f"Total subhalos:       {len(subhalos['SPOS'])}")
print(f"Position [0]:         {subhalos['SPOS'][0]}")
print(f"Stellar mass [0]:     {subhalos['SMST'][0, 4]:.4e}  (x1e10 Msun/h)")  # type 4 = stars
print(f"Half mass radius [0]: {subhalos['RHMS'][0]:.4f}  (ckpc/h)")            # total, not per type
print(f"Parent halo [0]:      {subhalos['GRNR'][0]}")

print()
print("=" * 60)
print("TEST 3: get_subhalo_particles_strict (ptype=4 stars)")
print("=" * 60)
sub_particles = get_subhalo_particles_strict(
    snap_base = SNAP_BASE,
    groups_dir = GROUPS_DIR,
    subhalo_idx = 0,
    ptype = 4,
    blocks = ['POS ', 'VEL ', 'MASS']
)
print(f"Stellar positions shape:  {sub_particles['POS '].shape}")
print(f"Stellar velocities shape: {sub_particles['VEL '].shape}")
print(f"Stellar masses shape:     {sub_particles['MASS'].shape}")
print(f"First star position:      {sub_particles['POS '][0]}")
print(f"First star mass:          {sub_particles['MASS'][0]:.4e}  (x1e10 Msun/h)")

print()
print("=" * 60)
print("TEST 4: get_halo_particles_strict (ptype=4 stars)")
print("=" * 60)
halo_particles = get_halo_particles_strict(
    snap_base = SNAP_BASE,
    groups_dir = GROUPS_DIR,
    halo_idx = 0,
    ptype = 4,
    blocks = ['POS ', 'VEL ', 'MASS']
)
print(f"Stellar positions shape:  {halo_particles['POS '].shape}")
print(f"Stellar velocities shape: {halo_particles['VEL '].shape}")
print(f"Stellar masses shape:     {halo_particles['MASS'].shape}")
print(f"First star position:      {halo_particles['POS '][0]}")
print(f"First star mass:          {halo_particles['MASS'][0]:.4e}  (x1e10 Msun/h)")

print()
print("=" * 60)
print("SANITY CHECK: halo >= subhalo particle count")
print("=" * 60)
print(f"Subhalo 0 stars: {sub_particles['POS '].shape[0]}")
print(f"Halo 0 stars:    {halo_particles['POS '].shape[0]}")
assert halo_particles['POS '].shape[0] >= sub_particles['POS '].shape[0], \
    "ERROR: halo should have >= particles than its central subhalo!"
print("PASSED: halo has >= stars than its central subhalo")

f = g3.GadgetFile('../data/magneticum/groups_120/sub_120.0', is_snap=False)
nsub = f.read('NSUB', 0)
ssub = f.read('SSUB', 0)
print('NSUB:', nsub)
print('SSUB:', ssub)