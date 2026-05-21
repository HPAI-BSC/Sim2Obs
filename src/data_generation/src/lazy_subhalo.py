import numpy as np
import pyarrow.dataset as ds
import pyarrow as pa

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubHalo
from src.data_generation.src.utils import setup_logger, rotate


logger = setup_logger("Lazy-Subhalo-Particles")


class LazySubhaloContext:
    def __init__(self, path, dataset=None) -> None:
        # store path so __add__ and other future logic can reference it safely
        self.path = path

        if dataset is None:
            self.dataset = ds.dataset(path, format="parquet")
        else:
            self.dataset = dataset

        self.data = None
        self.filters = []
        self.transforms = []

        # guard flags to prevent double-application of preprocess/rotate
        self._preprocessed = False
        self._rotated = False

    def add_filter(self, expr):
        self.filters.append(expr)

    def add_transform(self, fn):
        self.transforms.append(fn)

    def _build_filter(self):
        if not self.filters:
            return None

        expr = self.filters[0]
        for f in self.filters[1:]:
            expr = expr & f

        return expr

    def materialize(self):
        table = self.dataset.to_table(
            filter=self._build_filter()
        )

        for fn in self.transforms:
            table = fn(table)

        # assign to self.data so properties and __len__ work correctly
        self.data = table
        return table

    def get_subhalo_by_id(self, subhalo_id, halo_id=None) -> "SubHalo":
        # field names unified to "parent_id" / "subhalo_id" (consistent with schema)
        # pipeline filters and transforms are applied so results are consistent
        # with what materialize() would return
        base_filter = ds.field("subhalo_id") == subhalo_id
        if halo_id is not None:
            base_filter = (ds.field("parent_id") == halo_id) & base_filter

        pipeline_filter = self._build_filter()
        combined_filter = (pipeline_filter & base_filter) if pipeline_filter is not None else base_filter

        table = self.dataset.to_table(filter=combined_filter)

        for fn in self.transforms:
            table = fn(table)

        return table.to_pandas()

    def get_items_by_halo(self, parent_id) -> "pd.DataFrame":
        bucket = int(parent_id % 128)

        try:
            # parent_bucket_data is the in-file duplicate of the partition column
            base_filter = (
                    (ds.field("parent_bucket_data") == bucket) &
                    (ds.field("parent_id") == parent_id)
            )
            pipeline_filter = self._build_filter()
            combined_filter = (pipeline_filter & base_filter) if pipeline_filter is not None else base_filter

            table = self.dataset.to_table(filter=combined_filter)

            for fn in self.transforms:
                table = fn(table)

            return table.to_pandas()

        except Exception as e:
            logger.error("Failed to fetch items for halo %s (bucket %s): %s", parent_id, bucket, e)
            raise

    def filter_to_valid_halos(self, halo_context: "HaloContext"):
        halo_ids = halo_context.data["halo_id"].unique()
        buckets = np.unique(halo_ids % 128)

        logger.debug("Adding validity filter (lazy)")

        filter_expr = (
                ds.field("parent_bucket_data").isin(buckets.astype(int).tolist())
                &
                ds.field("parent_id").isin(halo_ids.astype(int).tolist())
        )

        self.filters.append(filter_expr)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        merged_dataset = ds.dataset(
            self.dataset.files + other.dataset.files,
            format="parquet"
        )
        # path=None is still passed but now clearly intentional since dataset is provided
        return self.__class__(path=None, dataset=merged_dataset)

    def preprocess(self, halo, lbox=None):
        # guard against double application
        if self._preprocessed:
            raise RuntimeError("preprocess() has already been called on this context.")
        self._preprocessed = True

        self.add_filter(ds.field("parent_id") == halo.halo_id)

        if hasattr(halo, "group_number"):
            self.add_filter(ds.field("group_number") == halo.group_number)

        # capture halo values eagerly at registration time, not by reference,
        # so mutations to `halo` after this call don't affect the transform
        gpos_snapshot = np.asarray(halo.gpos).copy()
        lbox_snapshot = lbox

        def pbc_transform(table):
            spos = np.asarray(table["spos"].to_pylist())
            delta = spos - gpos_snapshot  # use snapshot, not live halo

            if lbox_snapshot is not None:
                over = delta > lbox_snapshot / 2
                under = delta < -lbox_snapshot / 2
                delta[over] -= lbox_snapshot
                delta[under] += lbox_snapshot

            return table.set_column(
                table.schema.get_field_index("spos"),
                "spos",
                pa.array(delta.tolist())
            )

        self.add_transform(pbc_transform)

    def rotate(self, z, degree):
        # guard against double application
        if self._rotated:
            raise RuntimeError("rotate() has already been called on this context.")
        self._rotated = True

        # capture z and degree eagerly
        z_snapshot = z
        degree_snapshot = degree

        def rotate_fn(table):
            spos = np.asarray(table["spos"].to_pylist())
            rotated = rotate(z_snapshot, degree_snapshot, spos)

            return table.set_column(
                table.schema.get_field_index("spos"),
                "spos",
                pa.array(rotated.tolist())
            )

        self.add_transform(rotate_fn)

    @property
    def id_parents(self) -> pa.Array:
        # guard against uninitialized self.data
        if self.data is None:
            raise RuntimeError("Call materialize() before accessing id_parents.")
        return self.data["parent_id"]

    @property
    def subhalo_id(self) -> pa.Array:
        # guard against uninitialized self.data
        if self.data is None:
            raise RuntimeError("Call materialize() before accessing subhalo_id.")
        return self.data["subhalo_id"]

    def __len__(self):
        # self.data was always None; now materialize() sets it, so this works
        if self.data is None:
            raise RuntimeError("Call materialize() before calling len().")
        return self.data.num_rows


class LazySubhaloParticles:
    def __init__(self, path, dataset=None) -> None:
        self.path = path
        self.dataset = ds.dataset(path, format="parquet") if dataset is None else dataset

        self.data = None
        self.filters = []
        self.transforms = []

        self._preprocessed = False
        self._rotated = False

    # pipeline primitives
    def add_filter(self, expr):
        self.filters.append(expr)

    def add_transform(self, fn):
        self.transforms.append(fn)

    def _build_filter(self):
        if not self.filters:
            return None
        expr = self.filters[0]
        for f in self.filters[1:]:
            expr = expr & f
        return expr

    def materialize(self):
        table = self.dataset.to_table(filter=self._build_filter())

        for fn in self.transforms:
            table = fn(table)

        self.data = table
        return table

    # merge

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        merged_dataset = ds.dataset(
            self.dataset.files + other.dataset.files,
            format="parquet"
        )
        result = self.__class__(path=None, dataset=merged_dataset)

        # Mirror SubhaloParticles.__add__: track the boundary between self and other
        # so non_overlapping can slice back to just self's rows later
        self_rows = self.dataset.count_rows()
        result.overlapping_particles_id = self_rows

        return result

    # mirror SubhaloParticles.non_overlapping

    @property
    def non_overlapping(self):
        """
        Returns a new LazySubhaloParticles that, when materialized, contains
        only the rows that belonged to `self` before any __add__ merge.
        Mirrors SubhaloParticles.non_overlapping.
        """
        if not hasattr(self, "overlapping_particles_id"):
            logger.info("There are no overlapping particles, returning self.")
            return self

        boundary = self.overlapping_particles_id

        # Carry existing filters and transforms; append a row-slice transform
        result = self.__class__(path=None, dataset=self.dataset)
        result.filters = list(self.filters)
        result.transforms = list(self.transforms)

        result.add_transform(lambda table: table.slice(0, boundary))
        return result

    # mirror SubhaloParticles.preprocess

    def preprocess(self, halo, lbox=None):
        """
        Lazily registers a PBC position-normalisation transform on star_p,
        mirroring SubhaloParticles.preprocess.
        """
        if self._preprocessed:
            raise RuntimeError("preprocess() has already been called on this context.")
        self._preprocessed = True

        self.add_filter(ds.field("parent_id") == halo.halo_id)

        if hasattr(halo, "group_number"):
            self.add_filter(ds.field("group_number") == halo.group_number)

        # Snapshot eagerly — same fix as LazySubhaloContext.preprocess
        gpos_snapshot = np.asarray(halo.gpos).copy()
        lbox_snapshot = lbox

        def pbc_transform(table):
            star_p = np.array(table["star_p"].to_pylist())   # (N, 3)
            delta = star_p - gpos_snapshot

            if lbox_snapshot is not None:
                corrected = delta.copy()
                over = corrected > lbox_snapshot / 2
                under = corrected < -lbox_snapshot / 2
                corrected[over] -= lbox_snapshot
                corrected[under] += lbox_snapshot

                num_corrected = np.sum(over | under, axis=0)
                if np.sum(num_corrected) > 0:
                    logger.debug(
                        "Particles corrected due to PBC (x/y/z): %s / %s, halo: %s",
                        num_corrected, len(delta), halo.halo_id
                    )
                delta = corrected

            return table.set_column(
                table.schema.get_field_index("star_p"),
                "star_p",
                pa.array(delta.tolist())
            )

        self.add_transform(pbc_transform)

    # mirror SubhaloParticles.rotate

    def rotate(self, z, degree):
        """
        Lazily registers a rotation transform on star_p,
        mirroring SubhaloParticles.rotate.
        """
        if self._rotated:
            raise RuntimeError("rotate() has already been called on this context.")
        self._rotated = True

        z_snapshot = np.asarray(z).copy()
        degree_snapshot = degree

        def rotate_fn(table):
            star_p = np.array(table["star_p"].to_pylist())
            rotated = rotate(z_snapshot, degree_snapshot, star_p)
            return table.set_column(
                table.schema.get_field_index("star_p"),
                "star_p",
                pa.array(rotated.tolist())
            )

        self.add_transform(rotate_fn)

    # mirror SubhaloParticles properties / dunder

    @property
    def is_empty(self):
        if self.data is None:
            return True
        return self.data.num_rows == 0

    def __len__(self):
        if self.data is None:
            raise RuntimeError("Call materialize() before calling len().")
        return self.data.num_rows

    def __repr__(self):
        if self.data is None:
            return f"LazySubhaloParticles(unmaterialized, filters={len(self.filters)}, transforms={len(self.transforms)})"
        lines = [f"{col}: shape=({self.data.num_rows},)" for col in self.data.schema.names]
        return " ".join(lines)