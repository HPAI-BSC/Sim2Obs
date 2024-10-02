import os
import shutil

import numpy as np             # For numerical operations
import pandas as pd            # For data manipulation using DataFrames
import joblib                  # For loading the pre-trained model (pickle)
import numpy.ma as ma
from dotenv import load_dotenv
from scipy import stats

from api.secret_manager import SecretManager
from api.storage import Storage
from utils import get_p_subhalo, get_halos, get_subhalos, circle, muse_bins_r, get_transform_matrix, rotate_3dvec

# SELECT SIMULATIONS AND SET CONSTANTS
H = 0.6774      # Hubble constant
snapnum = 84    # Snapshot number (specific time step in the simulation)
z = 0.2         # redshift (measure how much the universe has expanded)
scalf = 0.8333  # scaling factor to convert from "comoving coordinates" to physical coordinates at the redshift

# TODO update this to access the dataset stored in your pc
# Dataset can be downloaded from /home/jovyan/minio/datasets/raw/space/Illustris-3
basePath = '/Users/martabarrosoisidoro/Illustris-3/output'
ihalo = 4       # selecting particular halos and subhalos
isubhalo = 4

# Connect to minio for pushing data using space-read-write policy
load_dotenv()
access_key = os.environ.get('ACCESS_KEY')
secret_key = os.environ.get('SECRET_KEY')

storage = Storage(access_key=access_key, secret_key=secret_key)
secret_manager = SecretManager(storage=storage)

os.makedirs("data", exist_ok=True)
os.makedirs("ICL+BCG-list/TNG-300", exist_ok=True)
os.makedirs("maps/Labels", exist_ok=True)
os.makedirs("maps/Velocity_Dispersion", exist_ok=True)

# READING HALO
r200, m200, gpos, fsub = get_halos(basePath, snapnum)


r200 *= scalf       # applying scaling factor
m200 *= 1e10        # converting halo masses to solar masses
gpos *= scalf       # applying scaling factor to halo positions


#READING PARTICLES
m, p, v = get_p_subhalo(basePath, snapnum, isubhalo=fsub[isubhalo])

m *= 1e10               # converting particle masses to solar masses
p *= scalf              # applying scaling factor to particle positions
v *= np.sqrt(scalf)     # scale velocities

v_cm = np.average(v, axis=0)    # calculate the center-of-mass velocity (avg velocity)
p -= gpos[isubhalo]             # shift particle positions relative to the subhalo's position


# Assume v is your array of velocities, shape (n, m)
v_centered = v - v_cm  # Subtract the mean velocity from all vectors

#SAVING
particle_array = np.array([m, np.sqrt((p**2).sum(axis=1)), np.sqrt(((v-v_cm)**2).sum(axis=1))]).T
path = f"data/particles_subhalo{isubhalo}_snap{snapnum}.dat"
np.savetxt(path, particle_array, header="m [Msun/h], r [kpc/h], v - v_cm [km/s]")
secret_manager.upload(path, path)

halos_array = np.array([r200[isubhalo], m200[isubhalo], *gpos[isubhalo]])
path = f"data/haloprop_halo{isubhalo}_snap{snapnum}.dat"
np.savetxt(path, halos_array, header="r200 [kpc/h], m200 [msun/h], gpos[kpc/h]")
secret_manager.upload(path, path)

spos, grnr, shmr = get_subhalos(basePath, snapnum)
spos *= scalf
shmr *= scalf
spos -= gpos[isubhalo]
subhalos_array = np.array([*spos[grnr == ihalo].T, shmr[grnr == ihalo]])
path = f"data/subhalos_halo{isubhalo}_snap{snapnum}.dat"
np.savetxt(path, subhalos_array.T, header="m [Msun/h], pos [kpc/h]")
secret_manager.upload(path,  path)

# label_ICLandBCG
# ----------------------------------------------------------------------

# ## Pickling the best model (trained using Random Forest)
# The variables with "S" in their names represent the units used in the simulations.

# Load the best model identified by a GridSearch algorithm (likely a Random Forest model)
final_model = joblib.load('model/Best_Model-Random-3par.pkl')

# Define simulation parameters
sim = "TNG-300"                # Name of the simulation (TNG-300)
G = 4.30091e-6                 # Gravitational constant in units of kpc Msun^-1 (km/s)^2
iFOF = 4                       # Index of the Friends-of-Friends (FoF) halo group

# Define the path to save results
save_path = f'ICL+BCG-list/{sim}/'

# List of features to be used in the model
Features = ["distance", "velocity", 'm200']

# Define file paths for particle and halo properties
filepart = f"data/particles_subhalo{iFOF}_snap{snapnum}.dat"  # File containing particle data for the specified halo and snapshot
filehalo = f"data/haloprop_halo{iFOF}_snap{snapnum}.dat"   # File containing halo properties for the specified halo and snapshot


# Need to read and save particle data before.
# Load particle data: mass (m), radius (r), and velocity (modulo_v)
m, r, modulo_v = np.loadtxt(filepart, skiprows=1).T

# Need to read and save halo data before.
# Load halo properties: R200 (radius) and M200 (mass)
R200, M200, _, _, _ = np.loadtxt(filehalo, skiprows=1)


# Calculate the circular velocity at R200 (V200) using the formula V200 = sqrt(G * M200 / R200)
V200 = np.sqrt(G * M200 / R200)

# Create a DataFrame with features required by the model
# Features are: log-scaled distance (r/R200), scaled velocity (modulo_v/V200), and log-scaled halo mass (M200)
Data = pd.DataFrame({
    'distance': np.log10(r / R200),                      # Log-scaled distance
    'velocity': modulo_v / V200,                         # Scaled velocity
    'm200': [np.log10(M200 * 1e-14)] * len(modulo_v)     # Log-scaled halo mass (m200, constant for all particles)
})

# Replace any infinite values in the DataFrame with NaN, then drop all rows with NaN values
Data = Data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

# Create an array of particle indices (0 to len(r)-1)
IDS = np.linspace(0, len(r) - 1, len(r), dtype=int)

# Use the trained model to make predictions on the data
# The model predicts whether each particle belongs to BCG (label 1) or ICL (label 0)
final_predictions = final_model.predict(Data)

# Extract indices of particles predicted to be part of the BCG
ids_BCG = IDS[np.where(final_predictions == 1)[0]]

# Extract indices of particles predicted to be part of the ICL
ids_ICL = IDS[np.where(final_predictions == 0)[0]]

# Save the indices of BCG particles to a CSV file
df = pd.DataFrame(ids_BCG)
path = save_path + 'FoF' + str(iFOF) + '_' + str(snapnum) + 'BCG.csv'
df.to_csv(path, index=False, header=False)
secret_manager.upload(path, path)

# Save the indices of ICL particles to a separate CSV file
df = pd.DataFrame(ids_ICL)
path = save_path + 'FoF' + str(iFOF) + '_' + str(snapnum) + 'ICL.csv'
df.to_csv(path, index=False, header=False)
secret_manager.upload(path, path)

# create_maps
# ----------------------------------------------------------------------

# Example MUSE parameters at redshift z = 0.3
bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

# ---------------- DEFINE PARAMETERS FOR SIMULATIONS ------------------
ihalo = 4
orig_system = np.eye(3)  # Original coordinate system (identity matrix)

# Vectors for rotating the system (axes of rotation)
vectors = [(0, 0, 1), (0, 1, 0)]  # Rotation vectors (e.g., z-axis and y-axis)

# Paths for saving and loading data
save_path = "maps/"
ICL_list_path = "ICL+BCG-list/TNG-300/"

# Read label ICL and BCG
indices_ICL = pd.read_csv(ICL_list_path + 'FoF' + str(ihalo) + "_" + str(snapnum) + 'ICL.csv',
                          header=None)
indices_ICL.columns = ['IDS']

# create a masking array for BCG and ICL (BCG==0, ICL==1)
Type4 = np.zeros_like(m)
try:
    Type4[indices_ICL["IDS"]] = 1
except:
    pass

# Read particle and halo data: IDs, mass, positions, velocities (depends on the simulation)
# spos : position of the galaxies
# r200: halo radius
# shmr: stellar half mass radius in halo
# p: star particle position
# v: star particle velocity
# m: star particle mass

n = np.array(vectors[1], dtype=np.float32)

Q = get_transform_matrix(orig_system, n, normalized=False)
pos_rot = np.float32(rotate_3dvec(Q, p))
vel_rot = np.float32(rotate_3dvec(Q, v))
spos_rot = np.float32(rotate_3dvec(Q, spos))


# Creating maps

# For each subhalo, find the corresponding r200 from its parent halo
r200_subhalo = r200[grnr]  # Map subhalo group numbers to their corresponding halo radii

# Now find interlopers by comparing subhalo positions to their corresponding halo radii
interlopers_id = np.where((np.abs(spos_rot[:, 0]) < r) & (np.abs(spos_rot[:, 1]) < r) & (np.abs(spos_rot[:, 2]) < r200_subhalo))[0]

# removing first galaxy (central/BCG), since it's usually the central galaxy or BCG
interlopers_id = interlopers_id[1:]

# velocity dispersion
vel_disp, binx, biny, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 1], (vel_rot[:, 2]) ** 2,
                                              statistic="mean", bins=bins, range=ranging)
vel_disp = np.sqrt(vel_disp / 3)

# icl fraction
label1, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 1], Type4,
                                            statistic="sum", bins=bins, range=ranging)
label2, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 1], Type4,
                                            statistic="count", bins=bins, range=ranging)
label = label1 / label2

# hiding zero particles signal
label[np.isnan(label)] = 0
vel_disp[np.isnan(vel_disp)] = 0

# create interlopers mask
# maschera: A mask is created to identify and mask out interlopers. The position (h, k) and radius (rradius) of
# each interloper are used to mark the corresponding pixels in the mask. The circle function checks if pixels fall
# within the interloper’s radius.
maschera = np.zeros_like(label)

for i in interlopers_id:
    h = spos_rot[i, 0]
    k = spos_rot[i, 1]
    rradius = shmr[i]
    for cc, x in enumerate(np.linspace(-r, r, bins)):
        try:
            maschera[cc, np.where(circle(x, biny, h, k, rradius) <= 1)[0]] = 1
        except IndexError:  # catch the error
            pass

        # Masking interlopers
vel_disp_masked = ma.masked_array(vel_disp, mask=maschera).filled(fill_value=0)
label_masked = ma.masked_array(label, mask=maschera).filled(fill_value=0)

# first zoom

'''
To create the zoom, we have made the histograms for twice the length of each side and we zoom these histograms.
We do so by randomly extracting new centers in the images (xoprime,yoprime) such as that their size is within 15% of
the toal size of the image. In other words, we shift the center so that it is not perfectly centered by 15%. We then 
crop the histogram to retain an image of the size provided by MUSE.
'''
xo, yo = (int(bins / 2), int(bins / 2))
xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)
vel_disp_masked1 = vel_disp_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                   yoprime - int(bins / 4): yoprime + int(bins / 4)]
label_masked1 = label_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                yoprime - int(bins / 4): yoprime + int(bins / 4)]

# second zoom
xo, yo = (int(bins / 2), int(bins / 2))
xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)
vel_disp_masked2 = vel_disp_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                   yoprime - int(bins / 4): yoprime + int(bins / 4)]
label_masked2 = label_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                yoprime - int(bins / 4): yoprime + int(bins / 4)]

# third zoom
xo, yo = (int(bins / 2), int(bins / 2))
xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)
vel_disp_masked3 = vel_disp_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                   yoprime - int(bins / 4): yoprime + int(bins / 4)]
label_masked3 = label_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                yoprime - int(bins / 4): yoprime + int(bins / 4)]

NAME_TO_SAVE = save_path + f"Velocity_Dispersion/halo{ihalo}_snap{snapnum}"
np.savez(NAME_TO_SAVE + f"xyz-zoom1", vel_disp_masked1)
np.savez(NAME_TO_SAVE + f"xyz-zoom2", vel_disp_masked2)
np.savez(NAME_TO_SAVE + f"xyz-zoom3", vel_disp_masked3)

secret_manager.upload(NAME_TO_SAVE + f"xyz-zoom1.npz", NAME_TO_SAVE + f"xyz-zoom1.npz")
secret_manager.upload(NAME_TO_SAVE + f"xyz-zoom2.npz", NAME_TO_SAVE + f"xyz-zoom2.npz")
secret_manager.upload(NAME_TO_SAVE + f"xyz-zoom3.npz", NAME_TO_SAVE + f"xyz-zoom3.npz")

NAME_TO_SAVE = save_path + f"Labels/halo{ihalo}_snap{snapnum}"
np.savez(NAME_TO_SAVE + f"xyz-zoom1", label_masked1)
np.savez(NAME_TO_SAVE + f"xyz-zoom2", label_masked2)
np.savez(NAME_TO_SAVE + f"xyz-zoom3", label_masked3)

secret_manager.upload(NAME_TO_SAVE + f"xyz-zoom1.npz", NAME_TO_SAVE + f"xyz-zoom1.npz")
secret_manager.upload(NAME_TO_SAVE + f"xyz-zoom2.npz", NAME_TO_SAVE + f"xyz-zoom2.npz")
secret_manager.upload(NAME_TO_SAVE + f"xyz-zoom3.npz", NAME_TO_SAVE + f"xyz-zoom3.npz")

# Remove folders
shutil.rmtree('data')
shutil.rmtree('ICL+BCG-list')
shutil.rmtree('maps')

