import pyarrow.dataset as ds

from src.data_generation.src.subhalo import SubHalo


class LazySubhaloContext:
    def __init__(self, path) -> None:
        self.dataset = ds.dataset(path, format="parquet")

    def load_halo(self, parent_id):
        bucket = parent_id % 128

        table = self.dataset.to_table(
            filter=(
                    (ds.field("parent_bucket") == bucket) &
                    (ds.field("id_parents") == parent_id)
            )
        )
        return table.to_pandas()

    def get_subhalo_by_id(self, subhalo_id, halo_id=None) -> SubHalo:
        if halo_id is not None:
            table = self.dataset.to_table(filter=((ds.field("parent_id") == halo_id) &
                                                  (ds.field("subhalo_id") == subhalo_id)))
        else:
            table = self.dataset.to_table(filter=(ds.field("subhalo_id") == subhalo_id))
        df = table.to_pandas()
        return df
