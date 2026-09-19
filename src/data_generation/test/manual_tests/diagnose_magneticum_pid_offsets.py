"""
Standalone diagnostic for the "GOFF/GLEN contiguous-layout assumption no
longer holds" error raised by MagneticumFetcherV2._owners_for_chunk.

Run directly on the cluster (needs g3read + real Magneticum group-catalogue
files, no dependency on the rest of this repo):

    python diagnose_magneticum_pid_offsets.py

Path/snapnum are hardcoded below (PATH, SNAPNUM) — edit them there if needed.

What it checks
---------------
_stream_match (data_fetcher.py) walks the group-catalogue files in order and
advances a running cumulative offset by `len(f.read("PID ", 2))` for each
file, then uses that offset to decide which halo/subhalo owns which global
particle position. That only works if `len(pid_chunk)` for a file exactly
equals the number of particles the file's GOFF/GLEN values were computed
against. The reference implementation for this exact task (g3matcha.py,
by the g3read author) instead uses `f.header.npart[2]` for this — a strong
hint that the two can disagree. This script prints both, per file, plus the
catalogue-wide totals implied by GOFF/GLEN/SOFF/SLEN, so we can see exactly
where (if anywhere) they diverge before changing any code.
"""
import os

import g3read as g3
import numpy as np

# Catalogue root: contains groups_<SNAPNUM>/ and snapdir_<SNAPNUM>/.
PATH = "/gpfs/scratch/bsc70/hpai/storage/projects/space/datasets/magneticum"
SNAPNUM = "120"


def main():
    groups_dir = os.path.join(PATH, f"groups_{SNAPNUM}")
    group_base = os.path.join(groups_dir, os.path.basename(groups_dir).replace("groups", "sub"))
    print(f"group_base = {group_base}")
    print()

    # ── catalogue-wide halo/subhalo metadata (same read your fetcher does) ──
    goff = g3.read_new(group_base, ["GOFF"], -1, is_snap=False, multiple_files=True)["GOFF"]
    glen = g3.read_new(group_base, ["GLEN"], -1, is_snap=False, multiple_files=True)["GLEN"]
    soff = g3.read_new(group_base, ["SOFF"], -1, is_snap=False, multiple_files=True)["SOFF"]
    slen = g3.read_new(group_base, ["SLEN"], -1, is_snap=False, multiple_files=True)["SLEN"]

    goff = np.asarray(goff, dtype=np.int64)
    glen = np.asarray(glen, dtype=np.int64)
    soff = np.asarray(soff, dtype=np.int64)
    slen = np.asarray(slen, dtype=np.int64)

    n_halos = len(goff)
    n_subhalos = len(soff)
    print(f"n_halos = {n_halos}, n_subhalos = {n_subhalos}")

    is_sorted = np.all(np.diff(goff) >= 0)
    n_out_of_order = int(np.sum(np.diff(goff) < 0)) if not is_sorted else 0
    print(f"GOFF already ascending on-disk? {is_sorted}"
          + ("" if is_sorted else f"  ({n_out_of_order} adjacent-row inversions)"))
    print()

    theoretical_total_from_glen = int(glen.sum())
    theoretical_total_from_goff_end = int((goff + glen).max()) if n_halos else 0
    theoretical_total_from_slen = int(slen.sum())
    theoretical_total_from_soff_end = int((soff + slen).max()) if n_subhalos else 0

    print("Theoretical PID-array size implied by the catalogue metadata:")
    print(f"  sum(GLEN)        = {theoretical_total_from_glen}")
    print(f"  max(GOFF+GLEN)   = {theoretical_total_from_goff_end}   (should equal sum(GLEN) if halos partition with no gaps/overlaps)")
    print(f"  sum(SLEN)        = {theoretical_total_from_slen}   (<= sum(GLEN); subhalos may not cover every group particle)")
    print(f"  max(SOFF+SLEN)   = {theoretical_total_from_soff_end}   (should be <= max(GOFF+GLEN))")
    print()

    # ── walk the PID stream file by file, two ways ──────────────────────────
    print(f"{'file':>28} | {'len(read PID)':>13} | {'header.npart[2]':>16} | {'match':>5} | "
          f"{'cum(len)':>12} | {'cum(npart)':>12}")
    print("-" * 105)

    cumulative_by_len = 0
    cumulative_by_npart = 0
    first_divergence = None
    n_files = 0

    for group_file in g3.yield_all_files(group_base):
        f = g3.GadgetFile(group_file, is_snap=False)
        pid_chunk = f.read("PID ", 2)
        n_by_len = 0 if pid_chunk is None else len(pid_chunk)
        n_by_npart = int(f.header.npart[2])
        match = (n_by_len == n_by_npart)

        if not match and first_divergence is None:
            first_divergence = (group_file, n_by_len, n_by_npart, cumulative_by_len, cumulative_by_npart)

        print(f"{os.path.basename(group_file):>28} | {n_by_len:>13} | {n_by_npart:>16} | "
              f"{'yes' if match else 'NO':>5} | {cumulative_by_len:>12} | {cumulative_by_npart:>12}")

        cumulative_by_len += n_by_len
        cumulative_by_npart += n_by_npart
        n_files += 1
        del pid_chunk

    print("-" * 105)
    print(f"n_files = {n_files}")
    print(f"total PID count, summed via len(read):      {cumulative_by_len}")
    print(f"total PID count, summed via header.npart[2]: {cumulative_by_npart}")
    print()

    if first_divergence is not None:
        gf, nl, nn, cl, cn = first_divergence
        print(f"FIRST DIVERGENCE at {os.path.basename(gf)}: len(read)={nl} vs header.npart[2]={nn} "
              f"(cumulative offset at that point: by-len={cl}, by-npart={cn})")
    else:
        print("No divergence between len(read PID) and header.npart[2] in any file.")

    print()
    print("Comparing catalogue-wide totals against the two accounting methods:")
    print(f"  sum(GLEN)                    = {theoretical_total_from_glen}")
    print(f"  max(GOFF+GLEN)                = {theoretical_total_from_goff_end}")
    print(f"  total PID via len(read)      = {cumulative_by_len}   "
          f"(diff vs sum(GLEN): {cumulative_by_len - theoretical_total_from_glen})")
    print(f"  total PID via header.npart[2] = {cumulative_by_npart}   "
          f"(diff vs sum(GLEN): {cumulative_by_npart - theoretical_total_from_glen})")


if __name__ == "__main__":
    main()
