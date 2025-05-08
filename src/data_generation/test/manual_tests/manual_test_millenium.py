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







