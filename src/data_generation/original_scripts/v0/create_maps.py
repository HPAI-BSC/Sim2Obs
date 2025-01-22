import numpy.ma as ma
import numpy as np
import pandas as pd
import os.path
import sys
from scipy import stats


def circle(x, y, h, k, r):
    '''
    This function checks if a point (x, y) lies within a circle of radius r
    centered at (h, k). Note: Angles (A) need to be in radians.
    '''
    return ((x - h) ** 2 + (y - k) ** 2) / r ** 2


def find_bin(x, r, Nbin):
    '''
    Maps a 1D coordinate in physical space to its corresponding bin number.
    This is useful for creating histograms or 2D maps.
    '''
    delta_r = 2 * r / Nbin  # Width of each bin
    x_prime = x + r  # Shift x to the range [0, 2r]
    return (x_prime / delta_r).astype(int)  # Return bin index


# Function to calculate the number of bins and radius for MUSE observations
def muse_bins_r(fov, spatial_resolution):
    '''
    Calculates the number of bins and the radius for MUSE observations based on
    the field of view (fov) and spatial resolution (both in kpc).
    '''
    bins = int(fov / spatial_resolution)  # Number of bins along one axis
    r = fov / 2  # Radius of the field of view
    return bins, r


# Example MUSE parameters at redshift z = 0.3
bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

# ---------------- DEFINE PARAMETERS FOR SIMULATIONS ------------------
ihalo = [1]  # List of clusters to be processed
num1 = 70  # Snapshot number for the simulation
orig_system = np.eye(3)  # Original coordinate system (identity matrix)

# Vectors for rotating the system (axes of rotation)
vectors = [(0, 0, 1), (0, 1, 0)]  # Rotation vectors (e.g., z-axis and y-axis)

# Paths for saving and loading data
save_path = "/dataA02/asaro/imarini/DATA_UNet/DATA_VV/2Reff/"
simulations_path = "/dataA02/asaro/imarini/Dianoga/"
ICL_list_path = "/dataA02/asaro/imarini/ICL+BCG-list/ICL+BCG-10x/"

# Read label ICL and BCG
indices_ICL = pd.read_csv(ICL_list_path + 'FoF' + str(ihalo) + "_" + str(num1) + 'ICL.csv',
                          header=None)
indices_ICL.columns = ['IDS']

# create a masking array for BCG and ICL (BCG==0, ICL==1)
Type4 = np.zeros_like(m)
try:
    Type4[indices_ICL["IDS"]] = 1
except:
    pass

# Read particle and halo data: IDs, mass, positions, velocities (depends on the simulation)
# ....
# ....
# spos : position of the galaxies
# r200: halo radius
# shmr: stellar half mass radius in halo
# p: star particle position
# v: star particle velocity
# m: star particle mass


# Creating maps
# interlopers
interlopers_id = np.where((np.abs(spos[:, 0]) < r) & (np.abs(spos[:, 1]) < r)
                          & (np.abs(spos[:, 2]) < r200))[0]

# removing first galaxy (central/BCG)
interlopers_id = interlopers_id[1:]

# velocity dispersion
vel_disp, _, _, _ = stats.binned_statistic_2d(p[:, 0], p[:, 1], (v[:, 2]) ** 2,
                                              statistic="mean", bins=bins, range=ranging)
vel_disp = np.sqrt(vel_disp / 3)

# icl fraction
label1, _, _, _ = stats.binned_statistic_2d(p[:, 0], p[:, 1], Type4,
                                            statistic="sum", bins=bins, range=ranging)
label2, _, _, _ = stats.binned_statistic_2d(p[:, 0], p[:, 1], Type4,
                                            statistic="count", bins=bins, range=ranging)
label = label1 / label2

# hiding zero particles signal
label[np.isnan(label)] = 0
vel_disp[np.isnan(vel_disp)] = 0

# create interlopers mask
maschera = np.zeros_like(v_mean)
for i in interlopers_id:
    h = spos[i, 0]
    k = spos[i, 1]
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

NAME_TO_SAVE = save_path + f"Velocity_Dispersion/halo{ihalo}_snap{num1}"
np.savez(NAME_TO_SAVE + f"xyz-zoom1", vel_disp_masked1)
np.savez(NAME_TO_SAVE + f"xyz-zoom2", vel_disp_masked2)
np.savez(NAME_TO_SAVE + f"xyz-zoom3", vel_disp_masked3)

NAME_TO_SAVE = save_path + f"Labels/halo{ihalo}_snap{num1}"
np.savez(NAME_TO_SAVE + f"xyz-zoom1", label_masked1)
np.savez(NAME_TO_SAVE + f"xyz-zoom2", label_masked2)
np.savez(NAME_TO_SAVE + f"xyz-zoom3", label_masked3)





