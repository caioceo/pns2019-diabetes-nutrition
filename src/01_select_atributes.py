# %%

import pandas as pd
df = pd.read_csv("../data/raw/pns/pns2019.csv")

# %%

atributos_gerais = {
    "Q03001": 'diabetes',
    "P00103": 'peso_informado',
    "P00104": 'peso_final',
    "P00404": 'altura_final',
    'A009010': 'qualidade_agua', 
    }

atributos_frequencia_alimentar_semanal = {
    "P006": "feijao",
    "P00901": "legumes e verduras (exceto batata, mandioca, cara e inhame)",
    "P01101": "carne vermelha", #
    "P013": "frango/galinha", # Juntar dados? (fontes de proteina animal)
    "P015": "peixe", # 
    "P02001": "suco de caixinha/lata/pó",
    "P01601": "suco natural", # comentário abaixo
    "P018": "frutas", # frutas mais consumidas pelos brasileiros e seu indice glicemico/açucar?
    "P02002": "refrigerante", # zero ou normal? (fontes de açucar)
    "P023": "leite",
    "P02501": "doces",
    "P02602": "lanches rapidos",
    "P027": "bebidas alcoolicas",
}

frequencia_mensal_alimentos = {
    "P03201": "cinco ou mais doses de bebida alcoolica",
}

frequencia_diaria_alimentos = {
    "P01001": "frequencia diaria de legumes e verduras (P00901)",
    "P019": "frequencia diaria de frutas (P018)", 
    "P02801": "frequencia diaria de bebidas alcoolicas (P027)"
}

tipo_alimentos = {
    "P02101": "tipo de suco industrializado (P02001)",
    "P02102": "tipo refrigerante (P02002)",
    "P02401": "tipo leite",
}

atividade_fisica = {
    "P034": "atividade nos ultimos 3 meses",
    "P035": "frequencia semanal de atividade fisica", # Descobrir tempo de exercicio semanal
    
    "P03701": "horas de exercicio por dia", # converter para minutos e somar com P03702
    "P03702": "minutos de exercicio por dia" #
    
}

# Consequência - Informação coletada apenas para diabéticos

atributos_recomedacoes_medicas = {
    "Q046011": 'orientacao alimentar',
    "Q046017": 'orientacao evitar acucar',
    "Q046016": 'orientacao evitar massas'
    
}

# %%

atributos_juntos = {}
atributos_juntos.update(atributos_gerais)
atributos_juntos.update(atributos_frequencia_alimentar_semanal)
atributos_juntos.update(frequencia_mensal_alimentos)
atributos_juntos.update(frequencia_diaria_alimentos)
atributos_juntos.update(tipo_alimentos)
atributos_juntos.update(atividade_fisica)

atributos_selecionado = list(atributos_juntos.keys())
descricao_dados = list(atributos_juntos.values())
# %%

df_cleaned = df[atributos_selecionado]
df_cleaned.to_csv('../data/processed/01_pns2019_filtered.csv', index=False)

# %%

import tabulate 

descricao_dados = [desc.capitalize() for desc in descricao_dados]

df_dicionario = pd.DataFrame({
    "atributos": atributos_selecionado,
    "descricao": descricao_dados
})    

df_dicionario.to_markdown('../docs/pns_filtered_dictionary.md', index=False)

# %%
