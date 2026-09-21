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


def _read_raw_block_int64(f, group_file, name, n_entities):
    """
    Bypass g3read's own dtype resolution and read a block's raw on-disk
    bytes directly (via the block's `start`/`length`, which g3read always
    populates regardless of what dtype it guessed), then reinterpret them
    as int64. Only meaningful if `length == n_entities * 8` (i.e. the block
    really is 8 bytes/entity on disk) — the caller checks that first.
    """
    block = f.blocks[name]
    with open(group_file, "rb") as fd:
        fd.seek(block.start)
        raw = fd.read(block.length)
    data = np.frombuffer(raw, dtype=np.int64, count=n_entities)
    if f.endian != "=":
        data = data.byteswap()
    return data


def check_ondisk_width_and_test_int64_fix(group_base, theoretical_total_from_glen):
    """
    Answers directly: is GOFF/SOFF actually stored as 8 bytes/entity on disk
    (in which case forcing an int64 re-read would fix the overflow), or
    genuinely 4 bytes/entity (in which case no read-side dtype change can
    help — the high bits were never written, and offsets must instead be
    reconstructed from GLEN/SLEN cumulative sums)?

    Does this by reading each block's raw on-disk byte length directly from
    the file's own block table (`GadgetBlock.length`, populated regardless
    of what dtype g3read *guessed*) and dividing by the known entity count
    for that file (`header.npart[0]` for halos, `npart[1]` for subhalos) —
    this is independent of g3read's own dtype inference, so it's a direct
    on-disk fact, not another guess.
    """
    print("=" * 78)
    print("ON-DISK WIDTH CHECK — is GOFF/SOFF really 4 bytes/entity on disk,")
    print("or 8 bytes/entity but misread as int32 by g3read?")
    print("=" * 78)

    goff_widths = set()
    soff_widths = set()
    info_printed = False

    for group_file in g3.yield_all_files(group_base):
        f = g3.GadgetFile(group_file, is_snap=False)
        n_halos_file = int(f.header.npart[0])
        n_subhalos_file = int(f.header.npart[1])

        if not info_printed:
            print("INFO-table dtype declaration (the file's own metadata, not a g3read guess):")
            for name in ["GOFF", "GLEN", "SOFF", "SLEN"]:
                if f.info is not None and name in f.info:
                    print(f"  {name}: {f.info[name]}")
                else:
                    print(f"  {name}: not present in an INFO table (format-1 file, or block missing)")
            print()
            info_printed = True

        goff_block = f.blocks.get("GOFF")
        soff_block = f.blocks.get("SOFF")
        if goff_block is not None and n_halos_file > 0:
            goff_widths.add(goff_block.length / n_halos_file)
        if soff_block is not None and n_subhalos_file > 0:
            soff_widths.add(soff_block.length / n_subhalos_file)

    print(f"distinct GOFF bytes/halo seen across all files:    {sorted(goff_widths)}")
    print(f"distinct SOFF bytes/subhalo seen across all files: {sorted(soff_widths)}")
    print()

    def _verdict(name, widths):
        if widths == {4.0}:
            print(f"{name} is stored as a 4-byte (int32) integer ON DISK, consistently.")
            print(f"  => Re-reading with dtype=int64 CANNOT recover the missing high bits:")
            print(f"     the true {name} value was already truncated when this file was written.")
            print(f"     A dtype change alone will NOT fix the PID assignment for {name}.")
            return "4byte"
        elif widths == {8.0}:
            print(f"{name} is stored as an 8-byte (int64) integer on disk — g3read may be")
            print(f"  misreading it as int32. Testing a forced int64 re-read below...")
            return "8byte"
        else:
            print(f"{name} bytes/entity is INCONSISTENT across files ({sorted(widths)}) — needs closer look.")
            return "mixed"

    goff_verdict = _verdict("GOFF", goff_widths)
    print()
    soff_verdict = _verdict("SOFF", soff_widths)
    print()

    goff_fixed, glen_fixed = None, None

    if goff_verdict == "8byte":
        print("Forcing an int64 re-read of GOFF across all files and rebuilding max(GOFF+GLEN)...")
        goff_int64_chunks = []
        glen_int64_chunks = []
        for group_file in g3.yield_all_files(group_base):
            f = g3.GadgetFile(group_file, is_snap=False)
            n_halos_file = int(f.header.npart[0])
            if n_halos_file == 0:
                continue
            goff_int64_chunks.append(_read_raw_block_int64(f, group_file, "GOFF", n_halos_file))
            glen_int64_chunks.append(f.read("GLEN", 0))

        goff_fixed = np.concatenate(goff_int64_chunks).astype(np.int64)
        glen_fixed = np.concatenate(glen_int64_chunks).astype(np.int64)
        new_max = int((goff_fixed + glen_fixed).max())
        print(f"  max(GOFF+GLEN) after forced int64 re-read: {new_max}")
        print(f"  sum(GLEN) (ground truth):                  {theoretical_total_from_glen}")
        if new_max == theoretical_total_from_glen:
            print("  MATCH — forcing int64 at read time fully fixes GOFF. This is the correct, minimal fix.")
        else:
            print("  MISMATCH — forcing int64 does NOT reproduce the ground truth; do not use this fix as-is.")
            goff_fixed, glen_fixed = None, None
        print()

    if goff_verdict == "4byte":
        print("Since GOFF is genuinely 4-byte on disk, testing the alternative fix instead:")
        print("reconstructing true_goff as a running int64 cumsum of GLEN in on-disk row order.")
        goff = g3.read_new(group_base, ["GOFF"], -1, is_snap=False, multiple_files=True)["GOFF"]
        glen = g3.read_new(group_base, ["GLEN"], -1, is_snap=False, multiple_files=True)["GLEN"]
        glen = np.asarray(glen, dtype=np.int64)
        true_goff = np.concatenate(([0], np.cumsum(glen, dtype=np.int64)[:-1]))
        new_max = int((true_goff + glen).max())
        print(f"  max(true_goff+GLEN) after cumsum reconstruction: {new_max}")
        print(f"  sum(GLEN) (ground truth):                        {theoretical_total_from_glen}")
        if new_max == theoretical_total_from_glen:
            print("  MATCH — the cumsum(GLEN) reconstruction is exact. Use this instead of reading GOFF directly.")
            goff_fixed, glen_fixed = true_goff, glen
        else:
            print("  MISMATCH — even the reconstruction disagrees; on-disk halo row order may not be reliable either.")
        print()

    print("=" * 78)
    print()
    return goff_fixed, glen_fixed


def check_entity_file_spanning(file_boundaries, offsets, lengths, entity_name, label, max_examples=5, max_end=None):
    """
    _owners_for_chunk processes the group catalogue one file at a time and
    relies on np.clip to correctly stitch together entities (halos or
    subhalos) whose particles straddle a file boundary. This has never
    actually been confirmed against real data — this checks, for every
    entity, whether its first and last particle land in the same file or
    different files, and shows real examples of the latter.

    `file_boundaries` is a list of (start, end) cumulative-offset pairs, one
    per group file in on-disk file order (already validated as the true
    global particle-position order by the PID length checks above).
    `offsets`/`lengths` should be whichever GOFF/GLEN (or SOFF/SLEN) array
    is currently trustworthy for the full range being checked. When passing
    RAW (possibly int32-overflowed) values, set `max_end` to the largest
    known-safe cumulative offset (e.g. 2**31) so wrapped/corrupted entities
    past that point — which could spuriously look like valid small,
    non-spanning offsets — are excluded rather than silently diluting the
    stats.
    """
    file_starts = np.array([b[0] for b in file_boundaries], dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    valid = lengths > 0
    if max_end is not None:
        valid = valid & (offsets + lengths <= max_end)
    n_valid = int(valid.sum())

    ends_inclusive = offsets[valid] + lengths[valid] - 1
    start_file = np.searchsorted(file_starts, offsets[valid], side="right") - 1
    end_file = np.searchsorted(file_starts, ends_inclusive, side="right") - 1
    spans = end_file - start_file

    n_spanning = int(np.sum(spans > 0))
    n_span_multi = int(np.sum(spans > 1))
    max_span = int(spans.max()) if n_valid else 0

    print(f"[{label}] {entity_name}: {len(offsets)} total, {n_valid} with length>0")
    if max_end is not None:
        print(f"  (restricted to entities with offset+length <= {max_end}, to exclude int32-wrapped values)")
    print(f"  spanning >=2 files: {n_spanning}  ({100.0 * n_spanning / max(1, n_valid):.4f}%)")
    print(f"  spanning >=3 files: {n_span_multi}  (max files touched by one entity: {max_span + 1})")

    if n_spanning > 0:
        idx_valid = np.nonzero(valid)[0]
        example_idx = idx_valid[spans > 0][:max_examples]
        print(f"  examples (index, offset, length, start_file -> end_file):")
        for i in example_idx:
            sf = int(np.searchsorted(file_starts, offsets[i], side="right") - 1)
            ef = int(np.searchsorted(file_starts, offsets[i] + lengths[i] - 1, side="right") - 1)
            print(f"    {entity_name}[{i}]: offset={offsets[i]}, length={lengths[i]}, "
                  f"file {sf} -> file {ef}")
    print()


def sign_unwrap_int32(raw: np.ndarray) -> np.ndarray:
    """
    The catalogue's total particle count (~3.36B) is less than 2**32
    (~4.29B), so a value that overflowed signed int32 wraps exactly once:
    a true value >= 2**31, truncated to int32 and sign-extended back to
    int64 by numpy, comes back as `true_value - 2**32` (negative). Undoing
    that is just: if the raw (int64-cast) value is negative, add 2**32 back.

    This works per-element, independently — no row order, no cumulative
    sums, no parent/child relationship needed. That matters because the
    naive `local_offset = raw_SOFF - raw_GOFF[parent]` approach (subtracting
    two raw values assuming they wrap "together") FAILS whenever the 2**31
    boundary falls in the middle of a halo's particle range: the halo's own
    GOFF can still be just under 2**31 (unwrapped) while a subhalo sitting
    further into that same halo's range crosses 2**31 and wraps on its own.
    """
    raw = np.asarray(raw, dtype=np.int64)
    return np.where(raw < 0, raw + (1 << 32), raw)


def check_sign_unwrap_reconstruction(group_base, goff, glen, true_goff):
    """
    Validates sign_unwrap_int32 against both GOFF and SOFF:
    - GOFF: sign_unwrap_int32(raw_GOFF) must exactly equal the already
      cumsum(GLEN)-validated true_goff, for every halo.
    - SOFF: sign_unwrap_int32(raw_SOFF) must nest inside its parent halo's
      true range, for every subhalo (this is the check the GRNR-subtraction
      approach failed on one case — see module docstring history).
    """
    print("=" * 78)
    print("SIGN-UNWRAP RECONSTRUCTION CHECK — per-value `if raw<0: raw+2**32`")
    print("=" * 78)

    goff_raw = np.asarray(goff, dtype=np.int64)
    glen = np.asarray(glen, dtype=np.int64)
    goff_unwrapped = sign_unwrap_int32(goff_raw)
    n_goff_mismatch = int(np.sum(goff_unwrapped != true_goff))
    print(f"GOFF: sign-unwrap vs cumsum(GLEN)-validated true_goff — mismatches: {n_goff_mismatch}")
    if n_goff_mismatch == 0:
        print("  PASS — sign-unwrap reproduces true_goff exactly for every halo.")
    else:
        bad = np.nonzero(goff_unwrapped != true_goff)[0][:5]
        for h in bad:
            print(f"  MISMATCH halo[{h}]: raw={goff_raw[h]}, sign_unwrap={goff_unwrapped[h]}, "
                  f"cumsum_true={true_goff[h]}")
    print()

    soff_raw = np.asarray(g3.read_new(group_base, ["SOFF"], 1, is_snap=False, multiple_files=True)["SOFF"], dtype=np.int64)
    slen = np.asarray(g3.read_new(group_base, ["SLEN"], 1, is_snap=False, multiple_files=True)["SLEN"], dtype=np.int64)
    grnr = np.asarray(g3.read_new(group_base, ["GRNR"], 1, is_snap=False, multiple_files=True)["GRNR"], dtype=np.int64)

    n = len(soff_raw)
    valid_grnr = (grnr >= 0) & (grnr < len(goff_raw))
    n_invalid_grnr = int(np.sum(~valid_grnr))
    print(f"n_subhalos = {n}, invalid GRNR (out of halo-index range): {n_invalid_grnr}")

    soff_unwrapped = sign_unwrap_int32(soff_raw)
    ok_lower = soff_unwrapped[valid_grnr] >= true_goff[grnr[valid_grnr]]
    ok_upper = (soff_unwrapped[valid_grnr] + slen[valid_grnr]) <= (true_goff[grnr[valid_grnr]] + glen[grnr[valid_grnr]])
    n_bad_lower = int(np.sum(~ok_lower))
    n_bad_upper = int(np.sum(~ok_upper))

    print(f"subhalos with sign-unwrapped SOFF < parent true_goff:           {n_bad_lower}")
    print(f"subhalos with sign-unwrapped SOFF+SLEN > parent true_goff+GLEN: {n_bad_upper}")

    success = n_invalid_grnr == 0 and n_goff_mismatch == 0 and n_bad_lower == 0 and n_bad_upper == 0
    if success:
        print("PASS — sign-unwrap alone (no GRNR needed) correctly reconstructs SOFF for")
        print("       every subhalo. This is the fix to use for both GOFF and SOFF.")
    else:
        print("FAIL — sign-unwrap alone does not fully hold either; needs a closer look.")
        idx_valid = np.nonzero(valid_grnr)[0]
        bad_mask = (~ok_lower) | (~ok_upper)
        bad_subhalo_idx = idx_valid[bad_mask][:5]
        for s in bad_subhalo_idx:
            h = int(grnr[s])
            print(f"  subhalo[{s}]: raw_SOFF={soff_raw[s]}, sign_unwrap={soff_unwrapped[s]}, SLEN={slen[s]}")
            print(f"    parent halo[{h}]: true_GOFF={true_goff[h]}, GLEN={glen[h]}")

    print()
    print("=" * 78)
    print()
    return (soff_unwrapped, slen) if success else (None, None)


def check_pid_and_snapshot_id_dtype(group_base, snap_base):
    """
    The int32 overflow found in GOFF/SOFF affects CUMULATIVE OFFSETS, but
    the actual particle ID VALUES matched against them (the group
    catalogue's "PID " block, and the snapshot's "ID  " block used by
    _stream_match / _store_particles) are a separate concern: if either is
    ALSO genuinely stored as a 4-byte integer on disk while true ID values
    exceed 2**31, those IDs would be silently truncated at write time too —
    and unlike GOFF/SOFF, individual particle IDs have no reconstructible
    structure (no monotonic cumsum, no known parent range) to recover the
    lost bits from. That would be a more serious, structurally
    unfixable-from-this-file bug if it exists — this checks whether it does.

    Cheap: dtype/width comes from file metadata (INFO table, block byte
    length), and value range comes from the group-catalogue PID scan the
    caller already did (passed in), not a fresh full data read. The
    snapshot side only reads block metadata, not the (potentially huge)
    actual particle data.
    """
    print("=" * 78)
    print("PID/ID DTYPE CHECK — group-catalogue PID block and snapshot ID block")
    print("=" * 78)

    info_pid = None
    pid_widths = set()
    for group_file in g3.yield_all_files(group_base):
        f = g3.GadgetFile(group_file, is_snap=False)
        n_pid_file = int(f.header.npart[2])
        if info_pid is None and f.info is not None and "PID " in f.info:
            info_pid = f.info["PID "]
        pid_block = f.blocks.get("PID ")
        if pid_block is not None and n_pid_file > 0:
            pid_widths.add(pid_block.length / n_pid_file)

    print(f"group catalogue 'PID ' INFO declaration: {info_pid}")
    print(f"group catalogue 'PID ' bytes/entity seen across all files: {sorted(pid_widths)}")
    if pid_widths == {4.0}:
        print("  => 'PID ' is 4-byte on disk. If true particle IDs exceed 2**31, they are")
        print("     ALREADY truncated in this file — no read-side or reconstruction fix")
        print("     is possible; see the PID value-range check below for whether that's live.")
    elif pid_widths == {8.0}:
        print("  => 'PID ' is 8-byte on disk — safe for IDs beyond 2**31.")
    else:
        print(f"  => inconsistent width ({sorted(pid_widths)}) — needs a closer look.")
    print()

    snap_files = list(g3.yield_all_files(snap_base))
    if snap_files:
        f = g3.GadgetFile(snap_files[0], is_snap=True)
        print(f"snapshot file (metadata only, no particle data read): {snap_files[0]}")
        for ptype in range(6):
            npart = int(f.header.npart[ptype])
            if npart == 0:
                continue
            id_block = f.blocks.get("ID  ")
            if id_block is None:
                print(f"  ptype {ptype}: npart={npart}, no 'ID  ' block found in this file")
                continue
            print(f"  ptype {ptype}: npart={npart}, g3read-detected ID dtype={id_block.data_type}, "
                  f"{id_block.partlen} bytes/particle")
        print()

        # _store_particles() only ever processes STAR particles (ptype 4).
        # The group-catalogue PID scan found ~23% of entries with a
        # structured bit-flag pattern (extreme magnitude, near int64
        # min/max) — likely a boundary/tracer/other-species ID convention.
        # If that pattern never shows up among actual star IDs, it's inert
        # for the code path we care about. This reads ONE snapshot file's
        # ptype=4 "ID  " data (bounded size — this file has ~6M stars) to
        # check directly, rather than assume.
        REASONABLE_ABS_BOUND = 1 << 40
        if int(f.header.npart[4]) > 0:
            star_ids = g3.read_new(snap_files[0], ["ID  "], 4, multiple_files=False, do_flatten=False)[4]["ID  "]
            star_ids = np.asarray(star_ids, dtype=np.int64)
            n_star_extreme = int(np.sum((star_ids > REASONABLE_ABS_BOUND) | (star_ids < -REASONABLE_ABS_BOUND)))
            print(f"star (ptype=4) IDs in {os.path.basename(snap_files[0])}: {len(star_ids)}, "
                  f"min={int(star_ids.min())}, max={int(star_ids.max())}")
            print(f"  of which extreme (abs>2**40, same bucket as the group-catalogue finding): {n_star_extreme}")
            if n_star_extreme == 0:
                print("  => star IDs are all 'normal' — the extreme PID pattern found in the group")
                print("     catalogue does not affect star particles, which is what the pipeline")
                print("     actually matches against (_store_particles only processes stars).")
            else:
                print("  => star IDs ALSO show the extreme pattern — this DOES affect the pipeline,")
                print("     needs further investigation before assuming it's benign.")
        else:
            print(f"no star particles (ptype=4) in {snap_files[0]} — cannot check directly")
    else:
        print(f"no snapshot files found at {snap_base} — skipping snapshot ID check")

    print()
    print("=" * 78)
    print()


def main():
    groups_dir = os.path.join(PATH, f"groups_{SNAPNUM}")
    group_base = os.path.join(groups_dir, os.path.basename(groups_dir).replace("groups", "sub"))
    snapshot_dir = os.path.join(PATH, f"snapdir_{SNAPNUM}")
    snap_base = os.path.join(snapshot_dir, os.path.basename(snapshot_dir).replace("snapdir", "snap"))
    print(f"group_base = {group_base}")
    print(f"snap_base = {snap_base}")
    print()

    check_pid_and_snapshot_id_dtype(group_base, snap_base)

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

    # Cross-check that header.npart[0]/[1] (the per-file entity counts used
    # as the denominator in the on-disk-width check) actually sum to these
    # independently-read totals — the "4 bytes/halo" width measurement is
    # only as trustworthy as this denominator being correct.
    npart0_sum = 0
    npart1_sum = 0
    for group_file in g3.yield_all_files(group_base):
        f = g3.GadgetFile(group_file, is_snap=False)
        npart0_sum += int(f.header.npart[0])
        npart1_sum += int(f.header.npart[1])
    print(f"sum(header.npart[0] across files) = {npart0_sum}  (vs n_halos = {n_halos}, "
          f"{'MATCH' if npart0_sum == n_halos else 'MISMATCH'})")
    print(f"sum(header.npart[1] across files) = {npart1_sum}  (vs n_subhalos = {n_subhalos}, "
          f"{'MATCH' if npart1_sum == n_subhalos else 'MISMATCH'})")
    print()

    is_sorted = np.all(np.diff(goff) >= 0)
    n_out_of_order = int(np.sum(np.diff(goff) < 0)) if not is_sorted else 0
    print(f"GOFF already ascending on-disk? {is_sorted}"
          + ("" if is_sorted else f"  ({n_out_of_order} adjacent-row inversions)"))
    if not is_sorted:
        wrap_rows = np.nonzero(np.diff(goff) < 0)[0]
        for r in wrap_rows:
            print(f"  inversion at on-disk row {r} -> {r + 1}: "
                  f"GOFF[{r}]={goff[r]}, GLEN[{r}]={glen[r]}, GOFF[{r + 1}]={goff[r + 1]}, GLEN[{r + 1}]={glen[r + 1]}")
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

    goff_fixed, glen_fixed = check_ondisk_width_and_test_int64_fix(group_base, theoretical_total_from_glen)

    soff_fixed, slen_fixed = (None, None)
    if goff_fixed is not None:
        soff_fixed, slen_fixed = check_sign_unwrap_reconstruction(group_base, goff, glen, goff_fixed)
    else:
        print("(no validated GOFF correction available — skipping the sign-unwrap check)")
        print()

    # ── walk the PID stream file by file, two ways ──────────────────────────
    print(f"{'file':>28} | {'len(read PID)':>13} | {'header.npart[2]':>16} | {'match':>5} | "
          f"{'cum(len)':>12} | {'cum(npart)':>12}")
    print("-" * 105)

    cumulative_by_len = 0
    cumulative_by_npart = 0
    first_divergence = None
    n_files = 0
    file_boundaries = []
    pid_value_min = None
    pid_value_max = None

    # "Reasonable" here means: could plausibly be a sequential particle
    # index for a box this size (total grouped particles ~3.36e9, so even a
    # generous full-box upper bound stays well under 2**40 ~ 1.1e12).
    # Anything outside that — especially near the int64 min/max boundaries —
    # is almost certainly a bit-flag/encoding scheme, not a truncation bug,
    # and needs to be understood on its own terms before assuming any fix.
    REASONABLE_ABS_BOUND = 1 << 40
    n_pid_negative = 0
    n_pid_extreme = 0
    n_pid_normal = 0
    extreme_examples = []

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

        # Track the actual PID *value* distribution (as g3read decoded it,
        # cast to int64) — separate from the cumulative *offset* overflow
        # checked above. Bucket into negative / extreme-magnitude / normal
        # rather than just min/max, since min/max alone can't distinguish
        # "int32 wraparound noise" (mildly negative, near -2**31) from
        # "deliberate bit-flag encoding" (near the int64 min/max boundary).
        if n_by_len > 0:
            chunk_i64 = np.asarray(pid_chunk, dtype=np.int64)
            cmin, cmax = int(chunk_i64.min()), int(chunk_i64.max())
            pid_value_min = cmin if pid_value_min is None else min(pid_value_min, cmin)
            pid_value_max = cmax if pid_value_max is None else max(pid_value_max, cmax)

            neg_mask = chunk_i64 < 0
            # (chunk_i64 > B) | (chunk_i64 < -B) instead of np.abs(...) > B —
            # np.abs overflows silently on the minimum representable int64
            # value, and we're already right at that boundary in this data.
            extreme_mask = (chunk_i64 > REASONABLE_ABS_BOUND) | (chunk_i64 < -REASONABLE_ABS_BOUND)
            n_pid_negative += int(neg_mask.sum())
            n_pid_extreme += int(extreme_mask.sum())
            n_pid_normal += int((~extreme_mask).sum())

            if len(extreme_examples) < 10:
                idx = np.nonzero(extreme_mask)[0][:10 - len(extreme_examples)]
                for i in idx:
                    v = int(chunk_i64[i])
                    extreme_examples.append((os.path.basename(group_file), int(i), v, hex(v & 0xFFFFFFFFFFFFFFFF)))
            del chunk_i64

        file_boundaries.append((cumulative_by_len, cumulative_by_len + n_by_len))
        cumulative_by_len += n_by_len
        cumulative_by_npart += n_by_npart
        n_files += 1
        del pid_chunk

    print("-" * 105)
    print(f"n_files = {n_files}")
    print(f"total PID count, summed via len(read):      {cumulative_by_len}")
    print(f"total PID count, summed via header.npart[2]: {cumulative_by_npart}")
    print()
    print(f"PID VALUE range across all files (as decoded, cast to int64): "
          f"min={pid_value_min}, max={pid_value_max}")
    print(f"PID value buckets: normal (abs<=2**40)={n_pid_normal}, "
          f"extreme (abs>2**40)={n_pid_extreme}, of which negative={n_pid_negative}")
    print(f"  extreme fraction: {100.0 * n_pid_extreme / max(1, cumulative_by_len):.6f}%")
    if extreme_examples:
        print("  examples of extreme PID values (file, index-in-file, value, hex):")
        for gf, i, v, h in extreme_examples:
            print(f"    {gf}[{i}]: {v}  ({h})")
    if n_pid_extreme == 0:
        print("  All PID values are comfortably within a plausible sequential-ID range.")
    else:
        print("  These are NOT int32-wraparound noise (that would be mildly negative, near")
        print("  -2**31) — the magnitudes here sit near the int64 min/max boundary, which")
        print("  looks like a deliberate bit-flag/encoding scheme rather than corruption.")
        print("  Needs to be understood (e.g. what fraction of particles this affects, and")
        print("  whether it correlates with a known boundary/tracer particle type) before")
        print("  assuming any fix — do NOT treat this the same as the GOFF/SOFF bug.")
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
    print()

    # ── do halos/subhalos actually straddle group-file boundaries? ──────────
    print("=" * 78)
    print("MULTI-FILE SPANNING CHECK — does any halo/subhalo's particle range")
    print("actually cross a group-catalogue file boundary?")
    print("=" * 78)

    check_entity_file_spanning(
        file_boundaries, goff, glen, "halo",
        label="raw GOFF/GLEN (only valid below the overflow point)", max_end=2 ** 31,
    )
    if goff_fixed is not None:
        check_entity_file_spanning(
            file_boundaries, goff_fixed, glen_fixed, "halo", label="corrected GOFF/GLEN (valid catalogue-wide)"
        )
    else:
        print("(no validated GOFF correction available yet — skipping the catalogue-wide halo check)")
        print()

    check_entity_file_spanning(
        file_boundaries, soff, slen, "subhalo",
        label="raw SOFF/SLEN (only valid below the overflow point)", max_end=2 ** 31,
    )
    if soff_fixed is not None:
        check_entity_file_spanning(
            file_boundaries, soff_fixed, slen_fixed, "subhalo", label="corrected SOFF/SLEN (valid catalogue-wide)"
        )
    else:
        print("(no validated SOFF correction available — skipping the catalogue-wide subhalo check)")
        print()


if __name__ == "__main__":
    main()
