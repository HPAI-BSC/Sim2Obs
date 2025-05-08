import unittest

import numpy as np
import pandas as pd
import illustris_python as il


class TestMilleniumData(unittest.TestCase):
    def setUp(self):
        self.path = '../../../../local/data/datasets/millenium/MTNG-L31.25-270-A/output'
        self.snapnum = 237

    def test_load_halos(self):
        halo_fields = ['Group_M_Crit200', 'Group_R_Crit200', 'GroupFirstSub', 'GroupPos']
        halos = il.groupcat.loadHalos(self.path, self.snapnum, fields=halo_fields)
        print(len(halos['Group_M_Crit200']))

    def test_load_subhalos(self):
        subhalo_fields = ['SubhaloPos', 'SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloGroupNr']
        subhalos = il.groupcat.loadSubhalos(self.path, self.snapnum, fields=subhalo_fields)
        print(len(subhalos['SubhaloPos']))

    def test_load_particles(self):
        particle_fields = ['Coordinates', 'Velocities', 'Masses']
        halo_fields = ['GroupFirstSub']
        halos = il.groupcat.loadHalos(self.path, self.snapnum, fields=halo_fields)

        d = {}
        errors_id = set()
        for halo_id in halos:
            try:
                if halo_id != -1:
                    particles = il.snapshot.loadSubhalo(self.path, self.snapnum, halo_id, 'star',
                                                        fields=particle_fields)
                    rows = particles['Coordinates'].shape[0]
                    d[halo_id] = np.array(rows)
            except Exception as e:
                errors_id.add(halo_id)
                continue
        print()
        df = pd.DataFrame.from_dict(d, orient='index')
        df = df.sort_values(by=0, ascending=False)
        print(f'Halo ids with errors: {len(errors_id)}')

    def test_check_relations(self):
        halo_fields = ['Group_M_Crit200', 'Group_R_Crit200', 'GroupFirstSub']
        halos = il.groupcat.loadHalos(self.path, self.snapnum, fields=halo_fields)

        halos_df = pd.DataFrame(halos)
        halos_df['halo_id'] = np.arange(0, len(halos_df))
        halos_df['Group_M_Crit200'] = halos_df['Group_M_Crit200'] * 1e10
        filtered_halos = halos_df[halos_df['Group_M_Crit200'] > 1e13 * 0.6774]
        print(f'Number of filtered halos by mass: {filtered_halos["halo_id"].nunique()}')

        subhalo_fields = ['SubhaloHalfmassRadType', 'SubhaloMassType', 'SubhaloGroupNr']
        subhalos = il.groupcat.loadSubhalos(self.path, self.snapnum, fields=subhalo_fields)
        subhalos['SubhaloMassType'] = subhalos['SubhaloMassType'][:, 4]
        subhalos['SubhaloHalfmassRadType'] = subhalos['SubhaloHalfmassRadType'][:, 4]
        subhalos_df = pd.DataFrame(subhalos)
        subhalos_df['SubhaloMassType'] = subhalos_df['SubhaloMassType'] * 1e10 / 0.6774
        filtered_subhalos = subhalos_df[subhalos_df['SubhaloMassType'] > 1e10]
        print(f'Number of filtered subhalos by mass: {len(filtered_subhalos)}')

        id_parents = filtered_subhalos['SubhaloGroupNr'].unique()
        print(f'Number of halo parents: {len(id_parents)}')

        halo_ids = filtered_halos['halo_id'].unique()
        print(f'Number of halos: {len(halo_ids)}')

        diff = set(halo_ids) - set(id_parents)
        print(f'Number of halos with no subhalos linked: {len(diff)}')







