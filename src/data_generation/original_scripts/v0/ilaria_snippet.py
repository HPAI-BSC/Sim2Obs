import illustris_python as il
import numpy as np

# SELECT SIMULATIONS AND SET CONSTANTS

H = 0.6774      # Hubble constant
snapnum = 84    # Snapshot number (specific time step in the simulation)
z = 0.2         # redshift (measure how much the universe has expanded)
scalf = 0.8333  # scaling factor to convert from "comoving coordinates" to physical coordinates at the redshift
                # of interest
# /home/jovyan/minio/datasets/raw/space/Illustris-3
basePath = '/Users/martabarrosoisidoro/Illustris-3/output'
ihalo = 4       # selecting particular halos and subhalos
isubhalo = 4


# FUNCTIONS TO OPERATE
# Retrieve particle data for a specific halo (coordinates, velocities, masses)
def get_p_halo(basePath, snapnum, ihalo, fields=None):
    '''GET PARTICLE DATA IN HALO'''
    '''isubhalo needs to be the first subhalo'''
    if fields is None:

        fields = ["Coordinates", "Velocities", "Masses"]
        star_p = il.snapshot.loadHalo(basePath, snapnum, ihalo, "star", fields=fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"]
        m = star_p["Masses"]
        return m, p, v
    else:
        print("Non usato field")
        star_p = il.groupcat.loadHalo(basePath, snapnum, ihalo, fields=fields)
        print(star_p.shape)
        return star_p


# Retrieve particle data for a specific subhalo (coordinates, velocities, masses)
def get_p_subhalo(basePath, snapnum, isubhalo, fields=None):
    ''' GET PARTICLE DATA IN SUBHALO isubhalo needs  to be the first subhalo '''
    if fields is None:

        fields = ["Coordinates", "Velocities", "Masses"]
        star_p = il.snapshot.loadSubhalo(basePath, snapnum, isubhalo,  "star", fields= fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"]
        m = star_p["Masses"]
        return m, p, v
    else:
        print ("Non usato field")
        star_p = il.groupcat.loadSubhalos(basePath, snapnum, isubhalo, fields=fields)
        print (star_p.shape)
        return star_p


# Retrieve global halo properties like mass (m200), radius (r200) and position (gpos) and index
# of the first subhalo (fsub)
def get_halos(basePath, snapnum, fields=None):
    '''GET HALO DATA'''
    if fields is None:

        fields = ["Group_M_Crit200","Group_R_Crit200","GroupPos", "GroupFirstSub"]
        halos = il.groupcat.loadHalos(basePath, snapnum, fields= fields)
        r200 = halos["Group_R_Crit200"]
        m200 = halos["Group_M_Crit200"]
        gpos = halos["GroupPos"]
        fsub = halos["GroupFirstSub"]

        return r200,m200,gpos,fsub
    else: 
        halos = il.groupcat.loadHalos(basePath, snapnum, fields=fields)
        return halos


# Retrieve subhalo properties like position (spos), group number the subhalo belongs to (grnr) and
# stellar half-mass radius (shmr)
def get_subhalos(basePath, snapnum, fields=None):
    '''GET SUBHALO DATA'''
    if fields is None:

        fields = ["SubhaloPos","SubhaloGrNr", "SubhaloHalfmassRadType"]
        halos = il.groupcat.loadSubhalos(basePath, snapnum, fields= fields)
        spos = halos["SubhaloPos"]
        grnr = halos["SubhaloGrNr"]
        shmr = halos["SubhaloHalfmassRadType"][:,4] #stars
        return spos,grnr,shmr
    else: 
        halos = il.groupcat.loadSubhalos(basePath, snapnum, fields=fields)
        return halos


# READING HALO
r200, m200, gpos, fsub = get_halos(basePath,snapnum)


r200 *= scalf       # applying scaling factor
m200 *= 1e10        # converting halo masses to solar masses
gpos *= scalf       # applying scaling factor to halo positions


# READING PARTICLES
# TODO not sure why the name of those variables are repeated
m, p, v = get_p_halo(basePath, snapnum, ihalo=ihalo)
m, p, v = get_p_subhalo(basePath, snapnum, isubhalo=fsub[isubhalo])

m *= 1e10               # converting particle masses to solar masses
p *= scalf              # applying scaling factor to particle positions
v *= np.sqrt(scalf)     # scale velocities

v_cm = np.average(v, axis=0)    # calculate the center-of-mass velocity (avg velocity)
p -= gpos[isubhalo]             # shift particle positions relative to the subhalo's position


#SAVING 
particle_array = np.array([m,np.sqrt((p**2).sum(axis=1)),np.sqrt(((v-v_cm)**2).sum(axis=1))]).T

np.savetxt(f"data/particles_subhalo{isubhalo}_snap{snapnum}.dat", particle_array, header = "m [Msun/h], r [kpc/h], v - v_cm [km/s]")

halos_array = np.array([r200[isubhalo],m200[isubhalo],*gpos[isubhalo]])

np.savetxt(f"data/haloprop_halo{isubhalo}_snap{snapnum}.dat", halos_array, header = "r200 [kpc/h], m200 [msun/h], gpos[kpc/h]")

spos,grnr,shmr = get_subhalos(basePath,snapnum)
spos*=scalf
shmr*=scalf
spos-=gpos[isubhalo]
subhalos_array = np.array([*spos[grnr==ihalo].T,shmr[grnr==ihalo]])
np.savetxt(f"data/subhalos_halo{isubhalo}_snap{snapnum}.dat", subhalos_array.T, header = "m [Msun/h], pos [kpc/h]")