import g3read as g3
import numpy as np


# Local use
def get_halo_particles(snap_base, group_file, halo_idx, ptype=4,
                       blocks=['POS ', 'VEL ', 'MASS'], radius_key='RCRI'):
    """
    Get particles of a given halo using a spatial sphere cut.

    Parameters
    ----------
    snap_base  : base path to snapshot (e.g. 'snap_120', no file number)
    group_file : path to group catalog file (single file)
    halo_idx   : index of the halo
    ptype      : particle type (4=stars, 0=gas, 1=DM)
    blocks     : snapshot blocks to retrieve
    radius_key : 'RCRI'=R200c, 'RVIR'=Rvir, 'R200'=R200, 'R500'=R500
    """
    f = g3.GadgetFile(group_file, is_snap=False)

    centers = f.read('GPOS', 0)
    radii = f.read(radius_key, 0)

    center = centers[halo_idx]
    radius = radii[halo_idx]

    print(f"Halo {halo_idx}: center={center}, radius={radius:.4f}")

    particles = g3.read_particles_in_box(
        snap_base, center, radius, blocks, ptype
    )

    return particles


# Cluster (correct approach)
def get_halo_particles_strict(snap_base, group_base, halo_idx, ptype=4,
                               blocks=['POS ', 'VEL ', 'MASS']):
    """
    Get particles strictly assigned to a halo by Subfind, using particle IDs.
    Requires the full group catalog (not a single sub-file).

    Parameters
    ----------
    snap_base  : base path to snapshot (e.g. '/data/snap_120')
    group_base : base path to group catalog (e.g. '/data/sub_120')
    halo_idx   : index of the halo
    ptype      : particle type (4=stars, 0=gas, 1=DM)
    blocks     : snapshot blocks to retrieve
    """
    # --- Read full group catalog ---
    goff = g3.read_new(group_base, ['GOFF'], 0, is_snap=False)['GOFF']
    glen = g3.read_new(group_base, ['GLEN'], 0, is_snap=False)['GLEN']
    pid = g3.read_new(group_base, ['PID '], 0, is_snap=False)['PID ']

    # Slice PIDs belonging to this halo
    start = goff[halo_idx]
    end = start + glen[halo_idx]
    halo_pids = set(pid[start:end])

    # --- Read snapshot particle IDs and requested blocks ---
    snap_data = g3.read_new(snap_base, blocks + ['ID  '], ptype)
    snap_ids = snap_data['ID  ']

    # Find which snapshot particles belong to this halo
    mask = np.isin(snap_ids, list(halo_pids))

    # Apply mask to all requested blocks
    result = {block: snap_data[block][mask] for block in blocks}

    return result


def get_subhalo_particles_strict(snap_base, group_base, subhalo_idx, ptype=4,
                                  blocks=['POS ', 'VEL ', 'MASS']):
    """
    Get particles strictly assigned to a subhalo by Subfind, using particle IDs.
    Requires the full group catalog (not a single sub-file).

    Parameters
    ----------
    snap_base    : base path to snapshot (e.g. '/data/snap_120')
    group_base   : base path to group catalog (e.g. '/data/sub_120')
    subhalo_idx  : index of the subhalo
    ptype        : particle type (4=stars, 0=gas, 1=DM)
    blocks       : snapshot blocks to retrieve
    """
    # --- Read full group catalog (ptype=1 for subhalos) ---
    soff = g3.read_new(group_base, ['SOFF'], 1, is_snap=False)['SOFF']
    slen = g3.read_new(group_base, ['SLEN'], 1, is_snap=False)['SLEN']
    pid = g3.read_new(group_base, ['PID '], 1, is_snap=False)['PID ']

    # Slice PIDs belonging to this subhalo
    start = soff[subhalo_idx]
    end = start + slen[subhalo_idx]
    subhalo_pids = set(pid[start:end])

    # --- Read snapshot ---
    snap_data = g3.read_new(snap_base, blocks + ['ID  '], ptype)
    snap_ids = snap_data['ID  ']

    # Match IDs
    mask = np.isin(snap_ids, list(subhalo_pids))

    result = {block: snap_data[block][mask] for block in blocks}

    return result


snapbase = '../data/magneticum/snap_120.0'
groupbase = '../data/magneticum/sub_120.0'

# Read the FoF catalog
fof = g3.GadgetFile(groupbase, is_snap=False)

# List all blocks
for block in fof.blocks:
    print(block)

# HALOS (ptype=0)
halo_positions = fof.read('GPOS', 0)
Group_M_Crit200 = fof.read('MCRI', 0)   # M_200c
Group_R_Crit200 = fof.read('RCRI', 0)   # R_200c
GroupFirstSub = fof.read('FSUB', 0)   # index of first subhalo

# SUBHALOS (ptype=1)
SubhaloPos = fof.read('SPOS', 1)   # subhalo position
SubhaloGrNr = fof.read('GRNR', 1)   # parent halo index
SMST = fof.read('SMST', 1)   # shape: (N_subhalos, 6)
# TODO one dim ? this is wrong # shape: (N_subhalos, 6)
RHMS = fof.read('RHMS', 1)
# Type index 4 = stars
SubhaloMassType        = SMST[:, 4]   # stellar mass per subhalo


# Reading Stellar Particles from Snapshot
snap = g3.GadgetFile(snapbase)
# ptype=4 = stars
Coordinates = snap.read('POS ', 4)
Velocities = snap.read('VEL ', 4)
Masses = snap.read('MASS', 4)

# From the group catalog
GOFF = fof.read('GOFF', 0)   # offset: where halo i's particles start in snapshot
GLEN = fof.read('GLEN', 0)   # length: how many particles belong to halo i

# Linking Particles to Halos
halo_idx = 0
start = GOFF[halo_idx]
end = start + GLEN[halo_idx]

halo_star_pos = Coordinates[start:end]
halo_star_vel = Velocities[start:end]
halo_star_mass = Masses[start:end]

# Linking Particles for subhalos
SOFF = fof.read('SOFF', 1)   # offset for subhalo i
SLEN = fof.read('SLEN', 1)   # length for subhalo i

sub_idx = 0
start = SOFF[sub_idx]
end = start + SLEN[sub_idx]

sub_star_pos = Coordinates[start:end]
sub_star_vel = Velocities[start:end]
sub_star_mass = Masses[start:end]

# -----
# Usage
# This includes all stars spatially within R200c, not strictly Subfind-assigned ones
particles = get_halo_particles(
    snap_base=snapbase,
    group_file=groupbase,
    halo_idx=0,
    ptype=4,              # stars
    blocks=['POS ', 'VEL ', 'MASS'],
    radius_key='RCRI'          # R200c
)

print('Stellar positions shape:', particles['POS '].shape)
print('Stellar masses shape:',    particles['MASS'].shape)


# Usage - to be executed in the cluster
'''
particles = get_halo_particles_strict(
    snap_base='/data/snap_120',
    group_base='/data/sub_120',
    halo_idx=0,
    ptype=4,
    blocks=['POS ', 'VEL ', 'MASS']
)

print('Stellar positions:', particles['POS '].shape)
print('Stellar masses:',   particles['MASS'].shape)

# Usage
particles = get_subhalo_particles_strict(
    snap_base   = '/data/snap_120',
    group_base  = '/data/sub_120',
    subhalo_idx = 0,
    ptype       = 4,
    blocks      = ['POS ', 'VEL ', 'MASS']
)

print('Stellar positions:', particles['POS '].shape)
print('Stellar masses:',    particles['MASS'].shape)
'''