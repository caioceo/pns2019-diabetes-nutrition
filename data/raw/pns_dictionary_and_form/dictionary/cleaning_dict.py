#%%
import pandas as pd
df = pd.read_excel('dicionario_limpo.xlsx');
df['descrição'] = df['descrição'].str.strip()

#%%

import json
atributos = (
    df['descrição']
    .dropna()
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
    .unique()
    .tolist()
)

with open('atributos.json', 'w', encoding='utf-8') as f:
    json.dump(atributos, f, ensure_ascii=False, indent=2)
# %%
