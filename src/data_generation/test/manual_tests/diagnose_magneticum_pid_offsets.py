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

    goff_fixed, glen_fixed = check_ondisk_width_and_test_int64_fix(group_base, theoretical_total_from_glen)

    # ── walk the PID stream file by file, two ways ──────────────────────────
    print(f"{'file':>28} | {'len(read PID)':>13} | {'header.npart[2]':>16} | {'match':>5} | "
          f"{'cum(len)':>12} | {'cum(npart)':>12}")
    print("-" * 105)

    cumulative_by_len = 0
    cumulative_by_npart = 0
    first_divergence = None
    n_files = 0
    file_boundaries = []

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
        label="raw SOFF/SLEN (only valid below the overflow point; no validated correction yet)", max_end=2 ** 31,
    )


if __name__ == "__main__":
    main()
