import pandas as pd
import json

df = pd.read_excel('processed_original_dict.xlsx')

df['descrição'] = (
    df['descrição']
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)

df['cod_variavel'] = (
    df['cod_variavel']
    .astype(str)
    .str.strip()
)

atributos_desejados = [
    
]

atributos_desejados = [a.strip() for a in atributos_desejados]

df_filtrado = df[
    df['descrição'].isin(atributos_desejados)
].dropna(subset=['cod_variavel', 'descrição'])

dicionario_filtrado = dict(
    zip(df_filtrado['cod_variavel'], df_filtrado['descrição'])
)

with open('filtered_dict_by_columns.json', 'w', encoding='utf-8') as f:
    json.dump(dicionario_filtrado, f, ensure_ascii=False, indent=2)