import os
import pickle
import eagleSqlTools as sql
import illustris_python as il
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose

from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.subhalo import SubhaloContext, SubHalo
from src.data_generation.src.utils import get_connection

load_dotenv('../.env')

# Initialize Hydra and load configuration
with initialize(config_path="../../../configs"):
    cfg_illustris = compose(config_name="data_generation.yaml")
    cfg_eagle = compose(config_name="data_generation.yaml")

with initialize(config_path="../../../configs"):
    cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
    cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
    cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'
    cfg_eagle.catalog.simulation_code = 'RefL0050N0752'

attr = next((step for step in cfg_illustris.pipeline if step.name == 'data_loader'), None)

os.environ['PROJECT_ROOT'] = '../../..'

# Illustris objects
halos = il.groupcat.loadHalos(cfg_illustris.catalog.path + '/output', cfg_illustris.catalog.snapnum,
                              fields=["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"])
halo = None
halo_context = None

if len(halos) > 0:
    scalf = cfg_illustris.catalog.scalf,
    if halos['GroupPos'].shape[1] == 3:
        gpos = [halos['GroupPos'][i] * scalf for i in range(halos['GroupPos'].shape[0])]
        halos['GroupPos'] = gpos

    halos = pd.DataFrame.from_dict(halos)
    halos['Group_M_Crit200'] = halos['Group_M_Crit200'] * attr.solar_mass_convention  # 𝑀⊙/ℎ
    halos['Group_R_Crit200'] = halos['Group_R_Crit200'] * scalf
    halos = halos[halos['Group_M_Crit200'] > attr.halo_min_mass * attr.H]
    halos['halo_id'] = np.arange(len(halos))

    # keep the first halo
    halos = halos.iloc[0, :].to_frame().T
    column_map = {'halo_id': 'halo_id', 'r200': 'Group_R_Crit200', 'm200': 'Group_M_Crit200', 'gpos': 'GroupPos',
                  'fsub': 'GroupFirstSub'}

    halo_context = HaloContext(halos, column_map)
    halo = Halo(
        halo_id=halos.loc[0, 'halo_id'],
        r200=halos.loc[0, 'Group_R_Crit200'],
        m200=halos.loc[0, 'Group_M_Crit200'],
        gpos=halos.loc[0, 'GroupPos'],
        fsub=halos.loc[0, 'GroupFirstSub'])

    with open('data/illustris/halo_0.pkl', 'wb') as f:
        pickle.dump(halo, f)

    with open('data/illustris/halo_context_0.pkl', 'wb') as f:
        pickle.dump(halo_context, f)

fields = ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType", "SubhaloParent"]
subhalos = il.groupcat.loadSubhalos(cfg_illustris.catalog.path + '/output', cfg_illustris.catalog.snapnum,
                                    fields=fields)

if len(subhalos) > 0:
    if subhalos['SubhaloPos'].shape[1] == 3:
        gpos = [subhalos['SubhaloPos'][i] for i in range(subhalos['SubhaloPos'].shape[0])]
        subhalos['SubhaloPos'] = gpos

    subhalos["SubhaloHalfmassRadType"] = subhalos["SubhaloHalfmassRadType"][:, 4]  # stars
    subhalos["SubhaloMassType"] = subhalos["SubhaloMassType"][:, 4] * attr.solar_mass_convention / attr.H  # 𝑀⊙
    subhalo_id = np.arange(len(subhalos["SubhaloMassType"]))

    subhalos = pd.DataFrame(subhalos)
    subhalos['subhalo_id'] = subhalo_id
    subhalos = subhalos[subhalos['SubhaloGrNr'] == int(halo.halo_id)]

    column_map = {"subhalo_id": "subhalo_id", "spos": "SubhaloPos", "id_parents": "SubhaloGrNr",
                  "shmr": "SubhaloHalfmassRadType", "mstr": "SubhaloMassType"}

    subhalos = subhalos.iloc[0, :].to_frame().T
    subhalo_context = SubhaloContext(subhalos, cfg_illustris.catalog.scalf, column_map)

    any_subhalo = SubHalo(
        subhalo_id=subhalos.loc[0, 'subhalo_id'],
        id_parents=subhalos.loc[0, 'SubhaloGrNr'],
        spos=subhalos.loc[0, 'SubhaloPos'],
        shmr=subhalos.loc[0, 'SubhaloHalfmassRadType'],
        mstr=subhalos.loc[0, 'SubhaloMassType']
    )

    with open('data/illustris/subhalo_any_0.pkl', 'wb') as f:
        pickle.dump(any_subhalo, f)

    with open('data/illustris/subhalo_context_0.pkl', 'wb') as f:
        pickle.dump(subhalo_context, f)

# Eagle objects
halo = None
conn = get_connection()
query = f"""
        SELECT 
            FOF.GroupID AS halo_id,                          -- Unique identifier for each halo
            FOF.Group_M_Crit200 AS m200,                     -- Mass within R200 (critical density)
            FOF.Group_R_Crit200 AS r200,                     -- Radius within which density is 200 times the critical density
            FOF.GroupCentreOfPotential_x AS pos_x,          -- X coordinate of halo's center
            FOF.GroupCentreOfPotential_y AS pos_y,          -- Y coordinate of halo's center
            FOF.GroupCentreOfPotential_z AS pos_z,          -- Z coordinate of halo's center
            SH.SubGroupNumber AS subhalo_id,                 -- ID of the subhalo
            SH.GroupNumber AS group_number,                  -- ID FOF 
            SH.MassType_Star AS subhalo_mass                -- Mass of the subhalo
        FROM 
            {cfg_eagle.catalog.simulation_code}_FOF AS FOF -- Halo properties
        LEFT JOIN 
            (
                SELECT 
                    SH.GroupID,
                    SH.SubGroupNumber,
                    SH.GroupNumber, 
                    SH.MassType_Star
                FROM 
                    {cfg_eagle.catalog.simulation_code}_Subhalo AS SH
                WHERE 
                    SH.MassType_Star = (
                        SELECT MAX(SH2.MassType_Star) -- Ensure it's the largest subhalo by mass
                        FROM {cfg_eagle.catalog.simulation_code}_Subhalo AS SH2
                        WHERE SH2.GroupID = SH.GroupID AND SH.MassType_Star > {attr.subhalo_min_mass}
                        AND SH.SnapNum = {cfg_eagle.catalog.snapnum} 
                    )
            ) AS SH ON FOF.GroupID = SH.GroupID
        WHERE 
            FOF.Group_M_Crit200 > {attr.halo_min_mass}  -- Minimum halo mass threshold
            AND FOF.SnapNum = {cfg_eagle.catalog.snapnum} -- Snapshot for redshift ~0.2
        """
# Run the query
array = sql.execute_query(conn, query)
# Units of measure [m200=M⊙, r200=pkpc, gpos=cMpc]
df = pd.DataFrame(array)
unique_halo_ids = df['halo_id'].unique()

if len(df) > 0:
    df = df.sort_values(by='halo_id')
    # pkps -> kps / h
    scalf = cfg_eagle.catalog.scalf
    gpos = df.apply(lambda row: np.array([row['pos_x'] * scalf / attr.H,
                                          row['pos_y'] * scalf / attr.H,
                                          row['pos_z'] * scalf / attr.H]), axis=1)
    df['gpos'] = gpos
    df['r200'] = df['r200'] / (attr.H * scalf)    # pkps -> ckps / h
    df['m200'] = df['m200'] / attr.H    # 𝑀⊙/ℎ
    df['halo_id'] = df['halo_id'].astype(int)
    df = df[(df['subhalo_id'] == 0) & (df['group_number'] == 2)]
    column_map = {'halo_id': 'halo_id', 'r200': 'r200', 'm200': 'm200', 'gpos': 'gpos', 'fsub': 'subhalo_id'}
    halo_context = HaloContext(df, column_map)
    idx = df.index.values[0]
    halo = Halo(
        halo_id=df.loc[idx, 'halo_id'],
        r200=df.loc[idx, 'r200'],
        m200=df.loc[idx, 'm200'],
        gpos=df.loc[idx, 'gpos'],
        fsub=df.loc[idx, 'subhalo_id'])

    with open('data/eagle/halo_0.pkl', 'wb') as f:
        pickle.dump(halo, f)

    with open('data/eagle/halo_context_0.pkl', 'wb') as f:
        pickle.dump(halo_context, f)

conn = get_connection()
query = f"""
            SELECT
                FOF.GroupID AS halo_id, -- Halo identifier
                FOF.Group_M_Crit200 AS m200, -- Halo mass within R200
                SH.SubGroupNumber AS subhalo_id, -- Subgroup
                SH.GroupNumber AS group_number, -- Group FOF
                SH.MassType_Star AS total_stellar_mass, -- Total stellar mass in the subhalo
                SH.HalfMassRad_Star AS stellar_half_mass_radius, --  stellar half-mass radius in the subhalo
                SH.CentreOfPotential_x AS subhalo_pos_x, --  X coordinate of subhalo's center
                SH.CentreOfPotential_y AS subhalo_pos_y, --  Y coordinate of subhalo's center
                SH.CentreOfPotential_z AS subhalo_pos_z --  Z coordinate of subhalo's center
            FROM
                {cfg_eagle.catalog.simulation_code}_FOF AS FOF -- Halo properties
            JOIN
                {cfg_eagle.catalog.simulation_code}_Subhalo AS SH -- Subhalo properties
            ON 
                FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
            WHERE
                SH.MassType_Star > {attr.subhalo_min_mass} -- Minimum subhalo stellar mass threshold
                AND FOF.Group_M_Crit200 > {attr.halo_min_mass} -- Minimum halo mass threshold
                AND SH.SnapNum = {cfg_eagle.catalog.snapnum} -- Snapshot for redshift ~0.2
            ORDER BY
                SH.SubGroupNumber ASC -- Order results by subhalo id
            """
# Executing the join is important so we ensure that we are selecting subhalos that are child of valid halos
array = sql.execute_query(conn, query)
df = pd.DataFrame(array)
# Units of measure [avg_stellar_half_mass_radius=pkpc, total_stellar_mass=M⊙, spos=cMpc]
if len(df) > 0:
    df = df.sort_values(by=['group_number', 'subhalo_id'])

    # pkps -> kps / h
    spos = df.apply(lambda row: np.array([row['subhalo_pos_x'] / attr.H, row['subhalo_pos_y'] / attr.H,
                                          row['subhalo_pos_z'] / attr.H]), axis=1).values
    df['spos'] = spos
    df['stellar_half_mass_radius'] = df['stellar_half_mass_radius'].values / attr.H  # pkps -> ckps / h
    df = df[(df['group_number'] == 2) & (df['subhalo_id'] == 0)]
    column_map = {"subhalo_id": "subhalo_id", "spos": "spos", "id_parents": "halo_id",
                  "shmr": "stellar_half_mass_radius", "mstr": "total_stellar_mass", "group_number": "group_number"}
    subhalo_context = SubhaloContext(df, cfg_eagle.catalog.scalf, column_map)
    idx = df.index.values[0]
    any_subhalo = SubHalo(
        subhalo_id=df.loc[idx, 'subhalo_id'],
        id_parents=df.loc[idx, 'halo_id'],
        spos=df.loc[idx, 'spos'],
        shmr=df.loc[idx, 'stellar_half_mass_radius'],
        mstr=df.loc[idx, 'total_stellar_mass'],
        group_number=df.loc[idx, 'group_number']
    )

    with open('data/eagle/subhalo_any_0.pkl', 'wb') as f:
        pickle.dump(any_subhalo, f)

    with open('data/eagle/subhalo_context_0.pkl', 'wb') as f:
        pickle.dump(subhalo_context, f)