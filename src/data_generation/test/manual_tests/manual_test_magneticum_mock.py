"""
probe_halo0.py
==============
Cluster diagnostic — uses the real Hydra config stack to initialise the
fetcher exactly as the production pipeline does, then extracts all data
for a target halo.

Tweak the PARAMETERS block below before submitting to the cluster.

Output
------
    <catalog.path>/probe_output/
        halo<id>_info.txt          — summary: counts, timings, RAM
        halo<id>_subhalos.parquet  — subhalo rows for the chosen halo
        halo<id>_particles.parquet — star particles (if within threshold)
"""

import gc
import logging
import os
import sys
import time
import tracemalloc

import numpy as np
import pyarrow.dataset as pads

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── rootutils ─────────────────────────────────────────────────────────────────

#from rootutils import rootutils
# rootutils.setup_root(__file__, pythonpath=True)


from src.data_generation.src.data_fetcher import DataFetcherFactory


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS — edit here before running
# ══════════════════════════════════════════════════════════════════════════════

# Execution environment: "local" or "cluster"
# Controls which paths config is loaded (paths/local.yaml vs paths/gpfs.yaml)
ENV: str = "cluster"

# Set to an integer to fix a specific halo, or None to auto-pick the smallest
HALO_ID: int | None = None

# Maximum GLEN (particle count) allowed before skipping particle extraction.
# Set to None to extract regardless of size.
MAX_GLEN: int | None = 100_000

# ══════════════════════════════════════════════════════════════════════════════

_PATHS_CONFIG = {
    "local":   "paths/local.yaml",
    "cluster": "paths/gpfs.yaml",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def estimate_particle_bytes(glen: int, n_fields: int = 9) -> int:
    """glen x n_fields x 4 bytes  (pos3 + vel3 + mass + age + metallicity)."""
    return glen * n_fields * 4


def snapshot_ram(label: str) -> str:
    current, peak = tracemalloc.get_traced_memory()
    msg = f"[RAM {label}]  current={fmt_bytes(current)}  peak={fmt_bytes(peak)}"
    logger.info(msg)
    return msg


# ── config ────────────────────────────────────────────────────────────────────

# Absolute path to the configs directory — resolved from this file's location
# so the script works regardless of the working directory it is launched from.
# Adjust the number of parent steps to match your repo layout, e.g.:
#   this file lives at:  <repo>/src/data_generation/scripts/probe_halo0.py
#   configs live at:     <repo>/configs
#   => 3 x .parent steps needed
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = str(os.path.normpath(os.path.join(_THIS_DIR, "../..", "..", "..", "configs")))


def build_config():
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CONFIGS_DIR, job_name="probe_halo0")
    cfg = compose(config_name="data_generation.yaml")

    paths_config = _PATHS_CONFIG.get(ENV)
    if paths_config is None:
        raise ValueError(f"Unknown ENV={ENV!r}. Choose 'local' or 'cluster'.")
    local_config = compose(config_name=paths_config)
    cfg.paths    = local_config.paths
    cfg.catalog  = compose("catalog/magneticum.yaml")["catalog"]

    return cfg


# ── main ──────────────────────────────────────────────────────────────────────

def magneticum_sample_size():
    tracemalloc.start()
    t_start = time.perf_counter()

    # ── build fetcher ─────────────────────────────────────────────────────────
    logger.info("Building Hydra config...")
    cfg = build_config()

    required_keys = ["G"]
    data_loader_dict = next(
        (step for step in cfg.pipeline if step.name == "data_loader"), None
    )
    attr_dict = {
        key: data_loader_dict[key]
        for key in required_keys
        if key in data_loader_dict
    }

    logger.info("Initialising fetcher...")
    fetcher = DataFetcherFactory.get_data_fetcher(cfg.catalog, **attr_dict)
    snapshot_ram("after fetcher init")

    # ── load catalog cache ────────────────────────────────────────────────────
    logger.info("Loading catalog cache (headers only)...")
    t0 = time.perf_counter()
    fetcher._load_catalog_cache()
    logger.info(f"Cache loaded in {time.perf_counter() - t0:.1f}s")
    snapshot_ram("after catalog cache")

    n_halos = len(fetcher._glen)
    n_subhalos = len(fetcher._soff)
    logger.info(f"Halos: {n_halos:,}   subhalos: {n_subhalos:,}")

    # ── choose target halo ────────────────────────────────────────────────────
    glen = fetcher._glen

    if HALO_ID is not None:
        target_id = HALO_ID
        logger.info(f"Using fixed HALO_ID={target_id}")
    else:
        nonzero = glen > 0
        if not np.any(nonzero):
            logger.error("All halos have GLEN=0.")
            sys.exit(1)
        target_id = int(np.argmin(np.where(nonzero, glen, glen.max() + 1)))
        logger.info(
            f"Auto-selected smallest halo: id={target_id}, glen={glen[target_id]:,}"
        )

    target_glen = int(fetcher._glen[target_id])
    target_nsub = int(fetcher._nsub[target_id])
    target_fsub = int(fetcher._fsub[target_id])
    target_goff = int(fetcher._goff[target_id])
    size_est = estimate_particle_bytes(target_glen)

    summary = [
        f"=== Halo probe (halo_id={target_id}) ===",
        "",
        f"Catalog path   : {cfg.catalog.path}",
        f"Snapnum        : {cfg.catalog.snapnum}",
        "",
        f"Total halos    : {n_halos:,}",
        f"Total subhalos : {n_subhalos:,}",
        "",
        "--- Target halo ---",
        f"halo_id        : {target_id}",
        f"GLEN (n PIDs)  : {target_glen:,}",
        f"GOFF           : {target_goff:,}",
        f"NSUB           : {target_nsub}",
        f"FSUB           : {target_fsub}",
        f"Est. RAM       : {fmt_bytes(size_est)}",
    ]
    logger.info("\n" + "\n".join(summary))

    # ── subhalos ──────────────────────────────────────────────────────────────
    logger.info("Extracting subhalos...")
    t0 = time.perf_counter()

    subhalo_ctx = fetcher.get_subhalos()

    if hasattr(subhalo_ctx, "dataset"):
        subs_df = (
            subhalo_ctx.dataset
            .to_table(filter=pads.field("id_parents") == target_id)
            .to_pandas()
        )
    else:
        subs_df = subhalo_ctx.data[
            subhalo_ctx.data["id_parents"] == target_id
        ].copy()

    elapsed_sub = time.perf_counter() - t0
    ram_sub = snapshot_ram("after subhalo extraction")
    logger.info(f"Subhalos: {len(subs_df)} rows in {elapsed_sub:.1f}s")

    summary += [
        "",
        "--- Subhalo extraction ---",
        f"Rows           : {len(subs_df)}",
        f"Elapsed        : {elapsed_sub:.1f}s",
        f"Columns        : {list(subs_df.columns)}",
        f"RAM snapshot   : {ram_sub}",
    ]

    del subs_df
    gc.collect()

    # ── particles ─────────────────────────────────────────────────────────────
    if ENV == "local":
        logger.info("ENV=local — skipping particle extraction (no snapshot data locally).")
        summary += ["", "--- Particle extraction ---", "SKIPPED (ENV=local)"]

    elif target_glen == 0:
        logger.warning("GLEN=0 - skipping particle extraction.")
        summary += ["", "--- Particle extraction ---", "SKIPPED (GLEN=0)"]

    elif MAX_GLEN is not None and target_glen > MAX_GLEN:
        logger.warning(
            f"GLEN={target_glen:,} > MAX_GLEN={MAX_GLEN:,}. "
            f"Skipping. Set MAX_GLEN={target_glen} or MAX_GLEN=None to extract."
        )
        summary += [
            "",
            "--- Particle extraction ---",
            f"SKIPPED : GLEN={target_glen:,} > MAX_GLEN={MAX_GLEN:,}",
            f"Re-run  : set MAX_GLEN={target_glen} or MAX_GLEN=None",
        ]

    else:
        logger.info(
            f"Extracting particles for halo_id={target_id}  "
            f"(GLEN={target_glen:,}, est. {fmt_bytes(size_est)})..."
        )
        t0 = time.perf_counter()

        particle_ctx = fetcher.get_p_halo(halo=target_id)

        if hasattr(particle_ctx, "dataset"):
            parts_df = (
                particle_ctx.dataset
                .to_table(filter=pads.field("parent_id") == target_id)
                .to_pandas()
            )
        else:
            parts_df = particle_ctx.data[
                particle_ctx.data["parent_id"] == target_id
            ].copy()

        elapsed_part = time.perf_counter() - t0
        actual_ram = parts_df.memory_usage(deep=True).sum()
        ram_part = snapshot_ram("after particle extraction")

        logger.info(
            f"Particles: {len(parts_df):,} rows  "
            f"{fmt_bytes(actual_ram)} RAM  {elapsed_part:.1f}s"
        )

        summary += [
            "",
            "--- Particle extraction ---",
            f"Rows           : {len(parts_df):,}",
            f"RAM (df)       : {fmt_bytes(actual_ram)}",
            f"Elapsed        : {elapsed_part:.1f}s",
            f"Columns        : {list(parts_df.columns)}",
            f"RAM snapshot   : {ram_part}",
        ]

        del parts_df
        gc.collect()

    # ── final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary += [
        "",
        "--- Overall ---",
        f"Total elapsed  : {total_elapsed:.1f}s",
        f"Peak RAM       : {fmt_bytes(peak)}",
    ]

    logger.info("\n" + "\n".join(summary))
    logger.info(f"All done in {total_elapsed:.1f}s  peak RAM={fmt_bytes(peak)}")


if __name__ == "__main__":
    magneticum_sample_size()