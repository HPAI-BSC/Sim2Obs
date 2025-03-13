import os
import eagleSqlTools as sql
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose
from src.data_generation.src.utils import get_connection
from src.data_generation.src.data_fetcher import DataFetcherFactory

load_dotenv('../../.env')

if __name__ == '__main__':
    cfg_eagle = None
    simulations = ['RefL0050N0752', 'AGNdT9L0050N0752', 'RefL0100N1504']
    os.environ['PROJECT_ROOT'] = '../../..'
    for simulation_code in simulations:
        with initialize(config_path="../../../../configs"):
            cfg_eagle = compose(config_name="data_generation.yaml")
            local_config = compose(config_name="paths/local.yaml")
            cfg_eagle.paths = local_config.paths
            cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
            cfg_eagle.catalog.simulation_code = simulation_code
            cfg_eagle.catalog.path = f'../../../../local/data/datasets/eagle/{simulation_code}'

        required_keys = ['G']
        data_loader_dict = next((step for step in cfg_eagle.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        fetcher_eagle = DataFetcherFactory.get_data_fetcher(cfg_eagle.catalog, **attr_dict)

        conn = get_connection()

        query = fetcher_eagle._build_query(table='halo')
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        print(f'saving {simulation_code} halo data ...')
        df.to_csv(f'../../../../local/data/datasets/eagle/{simulation_code}/output/fof.csv', index=False)

        query = fetcher_eagle._build_query(table='subhalo')
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        print(f'saving {simulation_code} subhalo data ...')
        df.to_csv(f'../../../../local/data/datasets/eagle/{simulation_code}/output/subhalo.csv', index=False)


