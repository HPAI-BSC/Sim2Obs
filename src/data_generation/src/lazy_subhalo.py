import numpy as np
import pyarrow.dataset as ds
import pyarrow as pa

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubHalo, SubhaloContext
from src.data_generation.src.utils import setup_logger, rotate


logger = setup_logger("Lazy-Subhalo-Particles")


class LazySubhaloContext:
    def __init__(self, path, dataset=None) -> None:
        self.path = path

        if dataset is None:
            self.dataset = ds.dataset(path, format="parquet")
        else:
            self.dataset = dataset

        self.data = None
        self.filters = []
        self.transforms = []

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

        self.data = table
        return table

    def get_subhalo_by_id(self, subhalo_id, halo_id=None) -> "SubHalo":
        # FIX: subhalo catalog schema field is "id_parents", not "parent_id"
        # (see MagneticumFetcher._get_subhalos: df["parent_bucket"] =
        # df["id_parents"] % 128 -- "id_parents" is what's actually written).
        base_filter = ds.field("subhalo_id") == subhalo_id
        if halo_id is not None:
            base_filter = (ds.field("id_parents") == halo_id) & base_filter

        pipeline_filter = self._build_filter()
        combined_filter = (pipeline_filter & base_filter) if pipeline_filter is not None else base_filter

        table = self.dataset.to_table(filter=combined_filter)

        for fn in self.transforms:
            table = fn(table)

        return table.to_pandas()

    @property
    def is_empty(self):
        print(type(self.dataset))
        return self.dataset.count_rows() == 0

    def get_items_by_halo(self, id_parents) -> "SubHaloContext":
        bucket = int(id_parents % 128)

        try:
            # parent_bucket_data is the in-file duplicate of the partition column
            # FIX: "id_parents", not "parent_id" -- see get_subhalo_by_id above.
            base_filter = (
                    (ds.field("parent_bucket_data") == bucket) &
                    (ds.field("id_parents") == id_parents)
            )
            pipeline_filter = self._build_filter()
            combined_filter = (pipeline_filter & base_filter) if pipeline_filter is not None else base_filter

            table = self.dataset.to_table(filter=combined_filter)

            for fn in self.transforms:
                table = fn(table)

            df = table.to_pandas()
            return SubhaloContext.from_dataframe(df)

        except Exception as e:
            logger.error("Failed to fetch items for halo %s (bucket %s): %s", id_parents, bucket, e)
            raise

    def filter_to_valid_halos(self, halo_context: "HaloContext"):
        halo_ids = halo_context.data["halo_id"].unique()
        buckets = np.unique(halo_ids % 128)

        logger.debug("Adding validity filter (lazy)")

        # FIX: "id_parents", not "parent_id" -- see get_subhalo_by_id above.
        filter_expr = (
                ds.field("parent_bucket_data").isin(buckets.astype(int).tolist())
                &
                ds.field("id_parents").isin(halo_ids.astype(int).tolist())
        )

        self.filters.append(filter_expr)

    def remove_subhalos_belonging_unvalid_halos(self, halo_context: "HaloContext"):
        """
        Lazy equivalent of SubhaloContext.remove_subhalos_belonging_unvalid_halos.

        LazySubhaloContext doesn't populate self.data until materialize() is
        called, so this can't eagerly filter a DataFrame the way the eager
        version does. Instead it registers an 'id_parents in valid halo_ids'
        filter via add_filter() -- composes with any other filters already
        registered, and is only actually applied when the dataset is scanned
        (materialize(), get_items_by_halo, etc.), consistent with the rest of
        this class's lazy design.

        NOTE: this largely duplicates filter_to_valid_halos() above (same
        class, same purpose, different name/signature). Kept as a separate
        method to match SubhaloContext's method name for API parity;
        consider consolidating the two if that duplication becomes a
        maintenance issue.

        Before/after counts are logged for diagnostic parity with the eager
        version, using dataset.count_rows(filter=...) rather than a full
        materialize() -- counts matching rows via metadata/row-group
        pruning where possible, without loading actual column data.
        """
        halo_ids = list(halo_context.data["halo_id"].unique())

        existing_filter = self._build_filter()
        count_before = (
            self.dataset.count_rows(filter=existing_filter)
            if existing_filter is not None
            else self.dataset.count_rows()
        )
        logger.debug(f"Number of subhalos before filtering: {count_before}")

        # FIX: "id_parents", not "parent_id" -- see get_subhalo_by_id above.
        self.add_filter(ds.field("id_parents").isin(halo_ids))

        count_after = self.dataset.count_rows(filter=self._build_filter())
        logger.debug(f"Number of subhalos after filtering: {count_after}")

    def get_items_by_subhalo_ids(self, subhalo_ids) -> "pd.DataFrame":
        """
        Lazy equivalent of `subhalo_context.data.loc[subhalo_ids]` on the
        eager SubhaloContext -- returns only the rows whose subhalo_id is
        in the given list, as a FRESH, STANDALONE pandas DataFrame.

        IMPORTANT: this does NOT materialize into this context -- self.data
        is left completely untouched (still None if it was None before this
        call). The returned DataFrame is built from a local pyarrow Table
        that is never assigned to self.data anywhere in this method. Call
        this as many times as needed without it ever changing what
        self.materialize()/self.data would later return.

        Like get_items_by_halo, this is a ONE-SHOT read: it respects any
        filters/transforms already registered on self, but does NOT
        register anything new on self.filters either. It mirrors the eager
        `.loc[...]` call's semantics (an immediate, side-effect-free read),
        not the deferred-filter semantics of
        remove_subhalos_belonging_unvalid_halos / filter_to_valid_halos.

        NOTE on efficiency: unlike get_items_by_halo, this can't use the
        parent_bucket_data partition column to prune buckets up front --
        bucket is derived from id_parents (id_parents % 128), not
        subhalo_id, and a list of subhalo_ids can span arbitrarily many
        different parent halos/buckets. This scans the full dataset
        (filtered via subhalo_id predicate pushdown at the row-group
        level, but not partition-pruned). If the subhalo_ids all share a
        known parent halo, get_items_by_halo (or preprocess()) will be
        cheaper.

        Parameters
        ----------
        subhalo_ids : list | array-like
            The subhalo_id values to select.

        Returns
        -------
        pd.DataFrame
            One row per matching subhalo_id found in the dataset. Missing
            ids are silently absent from the result (same as pandas'
            `.loc[]` behavior for an Index -- no explicit KeyError raised
            here, unlike get_items_by_halo which raises). This DataFrame
            is independent of self -- mutating it has no effect on this
            context, and calling materialize() afterward is unaffected by
            having called this.
        """
        logger.debug('Calling get_items_by_subhalo_ids')
        if all(isinstance(x, tuple) for x in subhalo_ids):
            subhalo_ids = [pair[1] for pair in subhalo_ids]

        subhalo_ids = list(subhalo_ids)
        base_filter = ds.field("subhalo_id").isin(subhalo_ids)
        logger.debug(base_filter)
        pipeline_filter = self._build_filter()
        combined_filter = (pipeline_filter & base_filter) if pipeline_filter is not None else base_filter
        logger.debug(combined_filter)
        table = self.dataset.to_table(filter=combined_filter)
        logger.debug(f'Table: {type(table)}')
        for fn in self.transforms:
            table = fn(table)

        return table.to_pandas()

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        merged_dataset = ds.dataset(
            self.dataset.files + other.dataset.files,
            format="parquet"
        )
        return self.__class__(path=None, dataset=merged_dataset)

    def preprocess(self, halo, lbox=None):
        if self._preprocessed:
            raise RuntimeError("preprocess() has already been called on this context.")
        self._preprocessed = True
        self.add_filter(ds.field("id_parents") == halo.halo_id)
        if hasattr(halo, "group_number"):
            self.add_filter(ds.field("group_number") == halo.group_number)

        gpos_snapshot = np.asarray(halo.gpos).copy()
        lbox_snapshot = lbox

        def pbc_transform(table):
            logger.debug('Executing pbc transform')
            spos = np.asarray(table["spos"].to_pylist())
            logger.debug(f'spos: {type(spos)} {spos.shape}')
            logger.debug(f'gpos: {type(gpos_snapshot)} {gpos_snapshot.shape}')

            delta = spos - gpos_snapshot

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
        if self._rotated:
            raise RuntimeError("rotate() has already been called on this context.")
        self._rotated = True

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
        if self.data is None:
            raise RuntimeError("Call materialize() before accessing id_parents.")
        # FIX: was reading self.data["parent_id"], a field that doesn't
        # exist in this schema -- the property is named id_parents because
        # that's the actual column name.
        return self.data["id_parents"]

    @property
    def subhalo_id(self) -> pa.Array:
        if self.data is None:
            raise RuntimeError("Call materialize() before accessing subhalo_id.")
        return self.data["subhalo_id"]

    def __len__(self):
        # Doesn't require materialize() -- count_rows(filter=...) uses
        # dataset metadata/row-group pruning, so it's cheaper than forcing
        # a full materialize() just to get a count, AND it means len()
        # works the same way on both SubhaloContext (always has self.data)
        # and this lazy class, for code written generically against
        # either (e.g. get_generator calling len(subhalo_context) right
        # after registering a filter, with no materialize() in between).
        #
        # Safe because none of this class's registered transforms
        # (pbc_transform, rotate_fn) change row count -- they only
        # rewrite the 'spos' column's values -- so count_rows(filter=...)
        # gives exactly the same count materialize() would produce.
        if self.data is not None:
            return self.data.num_rows
        return self.dataset.count_rows(filter=self._build_filter())


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

    def __len__(self):
        if self.data is None:
            raise RuntimeError("Call materialize() before calling len().")
        return self.data.num_rows

    def __repr__(self):
        if self.data is None:
            return f"LazySubhaloParticles(unmaterialized, filters={len(self.filters)}, transforms={len(self.transforms)})"
        lines = [f"{col}: shape=({self.data.num_rows},)" for col in self.data.schema.names]
        return " ".join(lines)