import os
from dotenv import load_dotenv
from hydra import initialize, compose
import pandas as pd
import eagleSqlTools as sql

from src.data_generation.src.utils import get_connection

load_dotenv('../.env')


if __name__ == '__main__':
    cfg_eagle = None

    with initialize(config_path="../../../configs"):
        cfg_eagle = compose(config_name="data_generation.yaml")
        local_config = compose(config_name="paths/local.yaml")
        cfg_eagle.paths = local_config
        cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
        cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
        cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'

    os.environ['PROJECT_ROOT'] = '../../..'
    data_loader_dict = cfg_eagle.pipeline[0]
    halo_min_mass = data_loader_dict['halo_min_mass']
    subhalo_min_mass = data_loader_dict['subhalo_min_mass']

    conn = get_connection()

    # Halos data
    query = f"""
                    SELECT 
                        FOF.GroupID AS halo_id,                          -- Unique identifier for each halo
                        FOF.Group_M_Crit200 AS m200,                     -- Mass within R200 (critical density)
                        FOF.Group_R_Crit200 AS r200,                     -- Radius within which density is 200 times the critical density
                        FOF.GroupCentreOfPotential_x AS pos_x,          -- X coordinate of halo's center
                        FOF.GroupCentreOfPotential_y AS pos_y,          -- Y coordinate of halo's center
                        FOF.GroupCentreOfPotential_z AS pos_z,          -- Z coordinate of halo's center
                        SH.GroupNumber AS group_number,                 -- ID FoF, for linking particles
                        SH.SubGroupNumber AS subhalo_id,                 -- ID of the subhalo
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
                                    WHERE SH2.GroupID = SH.GroupID 
                                    AND SH.SnapNum = {cfg_eagle.catalog.snapnum} 
                                )
                        ) AS SH ON FOF.GroupID = SH.GroupID
                    WHERE 
                        FOF.SnapNum = {cfg_eagle.catalog.snapnum} -- Snapshot for redshift ~0.2
                    """
    array = sql.execute_query(conn, query)
    df = pd.DataFrame(array)
    print('Halo data')
    print(df['halo_id'].nunique())
    df_halo_filtered = df[df['m200'] >= halo_min_mass]
    print(df_halo_filtered['halo_id'].nunique())
    df_subhalo_filtered = df_halo_filtered[df_halo_filtered['subhalo_mass'] >= subhalo_min_mass]
    print(df_subhalo_filtered['halo_id'].nunique())

    # Subhalos data
    query = f"""
                SELECT
                    FOF.GroupID AS halo_id, -- Halo identifier
                    FOF.Group_M_Crit200 AS m200, -- Halo mass within R200
                    SH.GroupNumber AS group_number, -- ID FoF, for linking particles
                    SH.SubGroupNumber AS subhalo_id, -- Subgroup
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
                    SH.SnapNum = {cfg_eagle.catalog.snapnum} -- Snapshot for redshift ~0.2
                ORDER BY
                    SH.SubGroupNumber ASC -- Order results by subhalo id
                """
    # Executing the join is important so we ensure that we are selecting subhalos that are child of valid halos
    array = sql.execute_query(conn, query)
    df = pd.DataFrame(array)

    # Filter by masses
    print('Subhalo data')
    print(df['subhalo_id'].nunique())
    df_halo_filtered = df[df['m200'] >= halo_min_mass]
    print(df_halo_filtered['subhalo_id'].nunique())
    df_subhalo_filtered = df_halo_filtered[df_halo_filtered['total_stellar_mass'] >= subhalo_min_mass]
    print(df_subhalo_filtered['subhalo_id'].nunique())


