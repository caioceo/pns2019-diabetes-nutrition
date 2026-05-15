import pandas as pd

# Atributos do JSON
cols = ["P006","P00901","P01001","P01101","P013","P015","P018","P019","P01601","P00601","P00602","P00603","P00604","P00605","P00607","P00608","P00609","P00610","P00611","P00612","P00613","P02002","P02102","P02501","P00617","P00618","P00619","P00620","P00621","P00622","P00623","P00614","P00615","P00616","P02601","P02602","P023","P02401","P02001","P02101","C006","C008","C009","C011","V0026","V0031","V0001","VDD004A","P034","P035","P03701","P03702","P036","P038","P039","P040","P04001","P04101","P04102","P04501","P04502","VDM001","Q00201","Q00202","Q00401","Q00503","Q00601","Q00101","Q02901","Q03001","Q031","Q03201","W00101","W00102","W00201","W00202","W00203","P00103","P00104","W00103","P00403","P00404","N010","N011","N012","N013","N014","N015","N016","N017","P050","P051","P05401","P060","P027","P02801","P029","P03201","VDF003","VDF004","I00102","J012","VDE014","E017","M005010"]
import os
BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, "..", "data", "processed", "01_pns2019_filtered_by_age_and_diabetes.csv"))
cols_existentes = [c for c in cols if c in df.columns]
cols_faltando = [c for c in cols if c not in df.columns]

if cols_faltando:
    print(f"Colunas ausentes no CSV ({len(cols_faltando)}): {cols_faltando}")

df[cols_existentes].to_csv("data/processed/02_pns2019_selected_columns.csv", index=False)
print(f"Exportado com {len(cols_existentes)} colunas e {len(df)} linhas.")