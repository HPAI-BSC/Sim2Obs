from dotenv import load_dotenv
import pandas as pd
import eagleSqlTools as sql

from src.data_generation.src.utils import get_connection

load_dotenv('../../.env')

if __name__ == '__main__':
    snapnum = 26
    halo_min_mass = 1e13    # Msun
    subhalo_min_mass = 1e10 # Msun
    run_halos = True
    run_subhalos = True

    simulations = ['AGNdT9L0050N0752', 'RefL0050N0752', 'RefL0100N1504']
    conn = get_connection()

    for simulation_code in simulations:
        print(f'Running {simulation_code} ...')
        if run_halos:
            # Halos data
            query = f"""
                            SELECT 
                                FOF.GroupID AS GroupID,                          -- Unique identifier for each halo
                                FOF.Group_M_Crit200 AS Group_M_Crit200,                     -- Mass within R200 (critical density)
                                FOF.Group_R_Crit200 AS Group_R_Crit200,                     -- Radius within which density is 200 times the critical density
                                FOF.GroupCentreOfPotential_x AS GroupCentreOfPotential_x,          -- X coordinate of halo's center
                                FOF.GroupCentreOfPotential_y AS GroupCentreOfPotential_y,          -- Y coordinate of halo's center
                                FOF.GroupCentreOfPotential_z AS GroupCentreOfPotential_z,          -- Z coordinate of halo's center
                                SH.GroupNumber AS GroupNumber,                 -- ID FoF, for linking particles
                                SH.SubGroupNumber AS SubGroupNumber,                 -- ID of the subhalo
                                SH.MassType_Star AS MassType_Star                -- Mass of the subhalo
                            FROM 
                                {simulation_code}_FOF AS FOF -- Halo properties
                            LEFT JOIN 
                                (
                                    SELECT 
                                        SH.GroupID,
                                        SH.SubGroupNumber,
                                        SH.GroupNumber, 
                                        SH.MassType_Star
                                    FROM 
                                        {simulation_code}_Subhalo AS SH
                                    WHERE 
                                        SH.MassType_Star = (
                                            SELECT MAX(SH2.MassType_Star) -- Ensure it's the largest subhalo by mass
                                            FROM {simulation_code}_Subhalo AS SH2
                                            WHERE SH2.GroupID = SH.GroupID 
                                            AND SH.SnapNum = {snapnum} 
                                        )
                                ) AS SH ON FOF.GroupID = SH.GroupID
                            WHERE 
                                FOF.SnapNum = {snapnum} -- Snapshot for redshift ~0.2
                            """
            array = sql.execute_query(conn, query)
            df = pd.DataFrame(array)
            #df.to_csv(f'../../../../local/data/datasets/eagle/{simulation_code}/output/fof.csv', index=False)
            print(f'Halos ids: {df["GroupID"].nunique()}')
            df_halo_filtered = df[df['Group_M_Crit200'] >= halo_min_mass]
            print(f'Filtered halos: {df_halo_filtered["GroupID"].nunique()}')

        if run_subhalos:
            # Subhalos data
            query = f"""
                        SELECT
                            FOF.GroupID AS GroupID, -- Halo identifier
                            FOF.Group_M_Crit200 AS Group_M_Crit200, -- Halo mass within R200
                            SH.GroupNumber AS GroupNumber, -- ID FoF, for linking particles
                            SH.SubGroupNumber AS SubGroupNumber, -- Subgroup
                            SH.MassType_Star AS MassType_Star, -- Total stellar mass in the subhalo
                            SH.HalfMassRad_Star AS HalfMassRad_Star, --  stellar half-mass radius in the subhalo
                            SH.CentreOfPotential_x AS CentreOfPotential_x, --  X coordinate of subhalo's center
                            SH.CentreOfPotential_y AS CentreOfPotential_y, --  Y coordinate of subhalo's center
                            SH.CentreOfPotential_z AS CentreOfPotential_z --  Z coordinate of subhalo's center
                        FROM
                            {simulation_code}_FOF AS FOF -- Halo properties
                        JOIN
                            {simulation_code}_Subhalo AS SH -- Subhalo properties
                        ON 
                            FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
                        WHERE
                            SH.SnapNum = {snapnum} -- Snapshot for redshift ~0.2
                            AND FOF.SnapNum = {snapnum}
                        ORDER BY
                            SH.SubGroupNumber ASC -- Order results by subhalo id
                        """
            query_paper = f"""
                        SELECT
                            FOF.GroupID AS GroupID, -- Halo identifier
                            FOF.Group_M_Crit200 AS Group_M_Crit200, -- Halo mass within R200
                            SH.GroupNumber AS GroupNumber, -- ID FoF, for linking particles
                            SH.SubGroupNumber AS SubGroupNumber, -- Subgroup
                            SH.MassType_Star AS MassType_Star, -- Total stellar mass in the subhalo
                            SH.HalfMassRad_Star AS HalfMassRad_Star, --  stellar half-mass radius in the subhalo
                            SH.CentreOfPotential_x AS CentreOfPotential_x, --  X coordinate of subhalo's center
                            SH.CentreOfPotential_y AS CentreOfPotential_y, --  Y coordinate of subhalo's center
                            SH.CentreOfPotential_z AS CentreOfPotential_z --  Z coordinate of subhalo's center
                        FROM
                            {simulation_code}_FOF AS FOF, -- Halo properties
                            {simulation_code}_Subhalo AS SH -- Subhalo properties
                        WHERE
                            FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
                            AND SH.SnapNum = {snapnum} -- Snapshot for redshift ~0.2
                            AND FOF.SnapNum = SH.SnapNum  -- Join snapnums to speedup query
                        ORDER BY
                            SH.SubGroupNumber ASC -- Order results by subhalo id
                        """

            # Executing the join is important so we ensure that we are selecting subhalos that are child of valid halos
            array = sql.execute_query(conn, query)
            df = pd.DataFrame(array)

            array = sql.execute_query(conn, query_paper)
            df_paper = pd.DataFrame(array)
            #df_paper.to_csv(f'../../../../local/data/datasets/eagle/{simulation_code}/output/subhalo.csv',
            #                index=False)

            # Both dataframes have the same sample size, the second will run faster
            print('Subhalos_ids: ', df['SubGroupNumber'].nunique(), df_paper['SubGroupNumber'].nunique())

            df_filtered_i = df[df['MassType_Star'] > subhalo_min_mass]
            print(f'Filtered subhalos ids applying just subhalo conditions: {df_filtered_i["SubGroupNumber"].nunique()}')
            print(f'Number of rows: {len(df_filtered_i)}')

            df_filtered_ii = df[(df['Group_M_Crit200'] > halo_min_mass) & (df['MassType_Star'] > subhalo_min_mass)]
            print(f'Filtered subhalos ids applying halos conditions: {df_filtered_ii["SubGroupNumber"].nunique()}')
            print(f'Number of rows: {len(df_filtered_ii)}')
            print()

