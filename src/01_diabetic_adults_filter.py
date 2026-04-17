# %% 
import pandas as pd
df = pd.read_csv("../data/raw/pns2019.csv")

# %%
todas_doencas = ["Q00201","Q03001","Q060","Q06306","Q068","Q074","Q079","Q088","Q092","Q11006","Q11604","Q120","Q128","Q124","Q084"]
menos_doenca_alvo = ["Q00201","Q060","Q06306","Q068","Q074","Q079","Q088","Q092","Q11006","Q11604","Q120","Q128","Q124","Q084"]
# %%

df_alvo = df[(df['C008'] >= 30) & (df['C008'] <= 60)]
df_saudaveis = df_alvo[(df_alvo[todas_doencas]==2).all(axis=1)]
df_diabeticos = df_alvo[(df_alvo[menos_doenca_alvo] == 2).all(axis=1) & (df_alvo["Q03001"] ==1)]
df_final = pd.concat([df_saudaveis, df_diabeticos], ignore_index=True)

# %%
df_final.to_csv("../data/processed/01_pns2019_filtered_by_age_and_diabetes.csv", index=False)

# %%
