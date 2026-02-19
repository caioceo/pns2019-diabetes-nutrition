# %% 

import pandas as pd

# %%

df = pd.read_csv("../data/processed/01_pns2019_filtered.csv")
# %%

df_publico_alvo = df[(df['C008'] >= 30) & (df['C008'] <= 60) & ((df['Q03001'] == 1) | (df['Q03001'] == 2))]
# %%
print("Diabéticos ou não:", df_publico_alvo.shape)
print("Diabéticos:", df_publico_alvo[df_publico_alvo['Q03001'] == 1].shape)
print ("Não diabéticos:", df_publico_alvo[df_publico_alvo['Q03001'] == 2].shape)
# %%

# Diabéticos ou não: (48324, 31)
# Diabéticos: (3110, 31)
# Não diabéticos: (45214, 31)

df_publico_alvo.to_csv('../data/processed/02_pns2019_filtered_by_age_and_diabetes.csv', index=False)
# %%
