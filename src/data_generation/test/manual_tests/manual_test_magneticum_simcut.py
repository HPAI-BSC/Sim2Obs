import pandas as pd

from src.data_generation.src.utils import m500_to_m200, r500_to_r200

df_1 = pd.read_csv('../../../../local/data/datasets/magneticum_simcut/dataset_1.csv', header=0)
df_2 = pd.read_csv('../../../../local/data/datasets/magneticum_simcut/dataset_2.csv', header=0)

df = pd.concat([df_1, df_2])
df = df.sort_values(by='uid')
index_duplicated = df.duplicated()
tmp = df[index_duplicated]
tmp2 = df.drop_duplicates()

assert len(df) == len(tmp2)
assert len(tmp) == 0

df['m200_msolh'] = df['M500_msolh'].apply(m500_to_m200)
df['r200_kpch'] = df['r500_kpch'].apply(r500_to_r200)

df.to_csv('../../../../local/data/datasets/magneticum/fof_simcut.csv', index=False)
df.to_csv('../data/magneticum/fof_simcut.csv', index=False)

#  TODO review

H = 0.6774
m_cond = 1e13 * 0.6774
filter_df = df[df['m200_msolh'] > m_cond]
# Mass > (1e13 * H) M⊙/h

df.to_csv('../../../../local/data/datasets/magneticum/fof_simcut_filtered.csv', index=False)
df.to_csv('../data/magneticum/fof_simcut_filtered.csv', index=False)


