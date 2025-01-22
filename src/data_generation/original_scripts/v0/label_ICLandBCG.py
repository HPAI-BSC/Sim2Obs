#!/usr/bin/env python
# coding: utf-8
import pickle

# Import necessary libraries
import numpy as np             # For numerical operations
import os.path                 # For handling file paths
import pandas as pd            # For data manipulation using DataFrames
import joblib                  # For loading the pre-trained model (pickle)

#----------------------------------------------------------------------
# ## Pickling the best model (trained using Random Forest)

# The variables with "S" in their names represent the units used in the simulations.

# Load the best model identified by a GridSearch algorithm (likely a Random Forest model)
final_model = joblib.load('../../model/Best_Model-Random-3par.pkl')

# Define simulation parameters
sim = "TNG-300"                # Name of the simulation (TNG-300)
G = 4.30091e-6                 # Gravitational constant in units of kpc Msun^-1 (km/s)^2
iFOF = 4                       # Index of the Friends-of-Friends (FoF) halo group
snap = 84                      # Snapshot number of the simulation

# Define the path to save results
save_path = f'ICL+BCG-list/{sim}/'

# List of features to be used in the model
Features = ["distance", "velocity", 'm200']

# Define file paths for particle and halo properties
filepart = f"data/particles_subhalo{iFOF}_snap{snap}.dat"  # File containing particle data for the specified halo and snapshot
filehalo = f"data/haloprop_halo{iFOF}_snap{snap}.dat"   # File containing halo properties for the specified halo and snapshot

#-------------------------------------------------------------------------------------------------

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
df.to_csv(save_path + 'FoF' + str(iFOF) + '_' + str(snap) + 'BCG.csv', index=False, header=False)

# Save the indices of ICL particles to a separate CSV file
df = pd.DataFrame(ids_ICL)
df.to_csv(save_path + 'FoF' + str(iFOF) + '_' + str(snap) + 'ICL.csv', index=False, header=False)
