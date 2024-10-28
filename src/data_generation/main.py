import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


from utils import get_halos, get_p_subhalo, get_subhalos, label_particles, rotate_particles, create_maps, \
    mask_interlopers, create_random_zoom

if __name__ == '__main__':
    snapnum = 84  # Snapshot number (specific time step in the simulation)
    basePath = '/Users/martabarrosoisidoro/Illustris-3/output'
    scalf = 0.8333  # scaling factor to convert from "comoving coordinates" to physical coordinates at the redshift
    H = 0.6774  # Hubble constant
    mstr_min = 1e10
    halo_mass_min = 1e13 * H
    rotation_vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    r200, m200, gpos, fsub = get_halos(basePath, snapnum)
    print(f'Number of halos in the snapshot: {len(r200)}')
    r200 *= scalf   # applying scaling factor
    m200 *= 1e10    # Converting halo masses to solar masses over H (removing 10e10 and H)
    gpos *= scalf  # applying scaling factor to halo positions
    # fsub is the primary or central subhalo for each halo

    spos, grnr, shmr, mstr = get_subhalos(basePath, snapnum)
    print(f"Number of subhalos in the snapshot: {len(spos)}")
    spos *= scalf
    shmr *= scalf

    mstr *= 1e10 / H  # Converting to solar masses
    id_parents = grnr
    spos = spos - gpos[id_parents]
    print(f'Number of unique halo ids inside subhalos data: {len(np.unique(grnr))}')
    print()

    id_errors = []
    big_halos = 0
    n = len(r200)
    for idhalo in range(n):
        halo_mass = m200[idhalo]
        if halo_mass > halo_mass_min:
            big_halos += 1
            idsubhalo = fsub[idhalo]
            # Get mass, positions and velocities of the stars within a certain subhalo
            if idsubhalo != 4294967295:
                print(f'Halo: {idhalo}')
                print(f' Primary subhalo: {idsubhalo}')

                m_subhalo = mstr[idsubhalo]
                if m_subhalo > mstr_min:
                    try:
                        # Try to download everything again
                        m, p, v = get_p_subhalo(basePath, snapnum, isubhalo=idsubhalo)
                        print(f' Subhalo: {idsubhalo} has {len(m)} particles.')
                    except Exception as e:
                        print(' Truncation error.')
                        print()
                        id_errors.append(idsubhalo)
                        continue

                    m *= 1e10  # converting particle masses to solar masses, no div in H due to the RF model
                    p *= scalf  # applying scaling factor to particle positions
                    v *= np.sqrt(scalf)  # scale velocities
            
                    v_cm = np.average(v, axis=0)  # calculate the center-of-mass velocity (avg velocity)
            
                    # Defining a new origin
                    p = p - gpos[idhalo]

                    # Computing the Euclidean distance from the new origin to each particle
                    distance = np.sqrt((p ** 2).sum(axis=1))
                    # Computing the delta vel. from each particle in comparison to the mean-vel.
                    v_delta = np.sqrt(((v - v_cm) ** 2).sum(axis=1))
            
                    halo_radius = r200[idhalo]
                    halo_mass = m200[idhalo]
            
                    # Scale subhalo data based on halo and label particles of primary subhalo
                    _, df_icl = label_particles(distance, v_delta, halo_radius, halo_mass)

                    # Take position of subhalos within current halo
                    spos_h = pd.DataFrame(spos)[id_parents == idhalo]
                    print(f' There are {len(spos_h)} subhalos.')

                    for i in range(len(rotation_vectors)):
                        pos_rot, vel_rot, spos_rot = rotate_particles(p, v, spos_h, rotation_vectors[i])

                        # Create a masking array for BCG and ICL (BCG==0, ICL==1)
                        type4 = np.zeros_like(m)
                        try:
                            type4[df_icl[0]] = 1
                        except:
                            pass

                        label_vel_biny_projections, interlopers_id = create_maps(halo_radius, spos_rot, pos_rot,
                                                                                 vel_rot, type4)
                        projections = ['xy', 'yz', 'xz']
                        for i, (label_i, vel_disp_i, biny_i) in enumerate(label_vel_biny_projections):
                            projection = projections[i]
                            shmr_h = pd.DataFrame(shmr)[id_parents == idhalo]
                            label_masked, vel_disp_masked = mask_interlopers(label_i, vel_disp_i, spos_rot, shmr_h,
                                                                             interlopers_id, biny_i)
                            label_masked1, vel_disp_masked1 = create_random_zoom(label_masked, vel_disp_masked)
                            label_masked2, vel_disp_masked2 = create_random_zoom(label_masked, vel_disp_masked)
                            label_masked3, vel_disp_masked3 = create_random_zoom(label_masked, vel_disp_masked)

                            NAME_TO_SAVE = f"data/npz/velocity_dispersion/halo-{idhalo}_subhalo-{idsubhalo}_snap{snapnum}_{projection}_"
                            np.savez(NAME_TO_SAVE + f"xyz-zoom1", vel_disp_masked1)
                            np.savez(NAME_TO_SAVE + f"xyz-zoom2", vel_disp_masked2)
                            np.savez(NAME_TO_SAVE + f"xyz-zoom3", vel_disp_masked3)

                            NAME_TO_SAVE = f"data/npz/labels/halo-{idhalo}_subhalo-{idsubhalo}_snap{snapnum}_{projection}_"
                            np.savez(NAME_TO_SAVE + f"xyz-zoom1", label_masked1)
                            np.savez(NAME_TO_SAVE + f"xyz-zoom2", label_masked2)
                            np.savez(NAME_TO_SAVE + f"xyz-zoom3", label_masked3)
                            print()
                        else:
                            print(' Primary subhalo does not reach minimum mass.')
                            print()

    print(f'Number of subhalos with errors: {len(id_errors)}')
    print(f'Number of big halos: {big_halos}')



