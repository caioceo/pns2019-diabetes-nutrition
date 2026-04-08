import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

df = pd.read_csv("../data/processed/01_pns2019_filtered_by_age_and_diabetes.csv")

df = df.rename(columns={
            "C008": 'Idade',
            "Q03001": 'Diabetes',
            "P00104": 'Peso',
            "P00404": 'Altura',
            "V0001": 'Estado'
        }
    )

map_estados = {
    '11' :'Rondônia',
    '12' : 'Acre',
    '13' : 'Amazonas',
    '14' : 'Roraima',
    '15' : 'Pará',
    '16' : 'Amapá',
    '17' : 'Tocantins',
    '21' : 'Maranhão',
    '22' : 'Piaui',
    '23': 'Ceará',
    '24':'Rio Grande do Norte',
    '25' :'Paraiba',
    '26' :'Pernambuco',
    '27':'Alagoas',
    '28':'Sergipe',
    '29':'Bahia',
    '31':'Minas Gerais',
    '32':'Espirito Santo',
    '33':'Rio de Janeiro',
    '35':'São Paulo',
    '41':'Paraná',
    '42':'Santa Catarina',
    '43':'Rio Grande do Sul',
    '50':'Mato Grosso do Sul',
    '51':'Mato Grosso',
    '52':'Goiás',
    '53':'Distrito Federal',
}

map_regiao = {
    'Rondônia': 'Norte',
    'Acre': 'Norte',
    'Amazonas': 'Norte',
    'Roraima': 'Norte',
    'Pará': 'Norte',
    'Amapá': 'Norte',
    'Tocantins': 'Norte',
    'Maranhão': 'Nordeste',
    'Piaui': 'Nordeste',
    'Ceará': 'Nordeste',
    'Rio Grande do Norte': 'Nordeste',
    'Paraiba': 'Nordeste',
    'Pernambuco': 'Nordeste',
    'Alagoas': 'Nordeste',
    'Sergipe': 'Nordeste',
    'Bahia': 'Nordeste',
    'Minas Gerais': 'Sudeste',
    'Espirito Santo': 'Sudeste',
    'Rio de Janeiro': 'Sudeste',
    'São Paulo': 'Sudeste',
    'Paraná': 'Sul',
    'Santa Catarina': 'Sul',
    'Rio Grande do Sul': 'Sul',
    'Mato Grosso do Sul': 'Centro-Oeste',
    'Mato Grosso': 'Centro-Oeste',
    'Goiás': 'Centro-Oeste',
    'Distrito Federal': 'Centro-Oeste'
}

df["Estado"] = df["Estado"].astype(str).map(map_estados)
df["Regiao"] = df["Estado"].map(map_regiao)

df_saudaveis = df[df['Diabetes'] == 2]  
df_diabeticos = df[df['Diabetes'] == 1] 


estado_stats = pd.DataFrame({
    'Total_Entrevistados': df.groupby('Estado').size(),
    'Total_Diabeticos': df_diabeticos.groupby('Estado').size(),
    'Total_Saudaveis': df_saudaveis.groupby('Estado').size()
}).fillna(0)

estado_stats['Percentual_Diabeticos'] = (estado_stats['Total_Diabeticos'] / estado_stats['Total_Entrevistados']) * 100
estado_stats['Percentual_Saudaveis'] = (estado_stats['Total_Saudaveis'] / estado_stats['Total_Entrevistados']) * 100

estado_stats_sorted = estado_stats.sort_values('Percentual_Diabeticos', ascending=False)

print("="*80)
print("ESTATÍSTICAS POR ESTADO")
print("="*80)
print(estado_stats_sorted.round(2))
print("\n")

fig, ax = plt.subplots(figsize=(14, 8))
estado_stats_sorted['Total_Diabeticos'].plot(kind='barh', ax=ax, color='coral')
ax.set_xlabel('Número de Diabéticos', fontsize=12)
ax.set_ylabel('Estado', fontsize=12)
ax.set_title('Número Absoluto de Diabéticos por Estado', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(14, 8))
estado_stats_sorted['Percentual_Diabeticos'].plot(kind='barh', ax=ax, color='indianred')
ax.set_xlabel('Percentual de Diabéticos (%)', fontsize=12)
ax.set_ylabel('Estado', fontsize=12)
ax.set_title('Percentual de Diabéticos em Relação ao Total de Entrevistados por Estado', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

top10_estados = estado_stats_sorted.head(10)
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(top10_estados))
width = 0.35

bars1 = ax.bar(x - width/2, top10_estados['Total_Entrevistados'], width, 
label='Total Entrevistados', color='skyblue')
bars2 = ax.bar(x + width/2, top10_estados['Total_Diabeticos'], width, 
label='Diabéticos', color='coral')

ax.set_xlabel('Estado', fontsize=12)
ax.set_ylabel('Número de Pessoas', fontsize=12)
ax.set_title('Top 10 Estados com Maior Percentual de Diabéticos', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top10_estados.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


regiao_stats = pd.DataFrame({
    'Total_Entrevistados': df.groupby('Regiao').size(),
    'Total_Diabeticos': df_diabeticos.groupby('Regiao').size(),
    'Total_Saudaveis': df_saudaveis.groupby('Regiao').size()
}).fillna(0)

regiao_stats['Percentual_Diabeticos'] = (regiao_stats['Total_Diabeticos'] / regiao_stats['Total_Entrevistados']) * 100
regiao_stats['Percentual_Saudaveis'] = (regiao_stats['Total_Saudaveis'] / regiao_stats['Total_Entrevistados']) * 100

regiao_stats_sorted = regiao_stats.sort_values('Percentual_Diabeticos', ascending=False)

print("="*80)
print("ESTATÍSTICAS POR REGIÃO")
print("="*80)
print(regiao_stats_sorted.round(2))
print("\n")

fig, ax = plt.subplots(figsize=(10, 6))
regiao_stats_sorted['Percentual_Diabeticos'].plot(kind='bar', ax=ax, color='tomato')
ax.set_xlabel('Região', fontsize=12)
ax.set_ylabel('Percentual de Diabéticos (%)', fontsize=12)
ax.set_title('Percentual de Diabéticos por Região', fontsize=14, fontweight='bold')
ax.set_xticklabels(regiao_stats_sorted.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
regiao_stats_sorted['Total_Entrevistados'].plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('Região', fontsize=12)
ax.set_ylabel('Número de Entrevistados', fontsize=12)
ax.set_title('Total de Entrevistados por Região', fontsize=14, fontweight='bold')
ax.set_xticklabels(regiao_stats_sorted.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
regiao_stats_sorted[['Total_Diabeticos', 'Total_Saudaveis']].plot(
    kind='bar', ax=ax, color=['coral', 'lightgreen']
)
ax.set_xlabel('Região', fontsize=12)
ax.set_ylabel('Número de Pessoas', fontsize=12)
ax.set_title('Comparação: Diabéticos vs Saudáveis por Região', fontsize=14, fontweight='bold')
ax.set_xticklabels(regiao_stats_sorted.index, rotation=45, ha='right')
ax.legend(['Diabéticos', 'Saudáveis'])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
regiao_stats_sorted['Total_Diabeticos'].plot(
    kind='pie', ax=ax, autopct='%1.1f%%', colors=colors, startangle=90
)
ax.set_ylabel('')
ax.set_title('Distribuição de Diabéticos por Região', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


print("="*80)
print("ANÁLISE DETALHADA: ESTADOS POR REGIÃO")
print("="*80)

for regiao in regiao_stats_sorted.index:
    print(f"\n{'='*80}")
    print(f"REGIÃO: {regiao}")
    print(f"{'='*80}")
    
    estados_regiao = df[df['Regiao'] == regiao].groupby('Estado').size().sort_values(ascending=False)
    diabeticos_regiao = df_diabeticos[df_diabeticos['Regiao'] == regiao].groupby('Estado').size()
    
    regiao_detail = pd.DataFrame({
        'Total_Entrevistados': estados_regiao,
        'Total_Diabeticos': diabeticos_regiao
    }).fillna(0)
    
    regiao_detail['Percentual_Diabeticos'] = (regiao_detail['Total_Diabeticos'] / 
                                                regiao_detail['Total_Entrevistados']) * 100
    regiao_detail = regiao_detail.sort_values('Percentual_Diabeticos', ascending=False)
    
    print(regiao_detail.round(2))


print("\n" + "="*80)
print("RESUMO GERAL DA PESQUISA")
print("="*80)
print(f"Total de entrevistados: {len(df)}")
print(f"Total de diabéticos: {len(df_diabeticos)}")
print(f"Total de saudáveis: {len(df_saudaveis)}")
print(f"Percentual de diabéticos na amostra: {(len(df_diabeticos)/len(df))*100:.2f}%")
print(f"Percentual de saudáveis na amostra: {(len(df_saudaveis)/len(df))*100:.2f}%")
print("\n")

estado_stats_sorted.to_csv('../data/processed/estatisticas_por_estado.csv')
regiao_stats_sorted.to_csv('../data/processed/estatisticas_por_regiao.csv')
print("Estatísticas exportadas para CSV!")


print("\n" + "="*80)
print("TABELA DE FREQUÊNCIA GERAL - DIABETES")
print("="*80)

freq_diabetes = pd.DataFrame({
    'Categoria': ['Diabéticos', 'Saudáveis', 'Total'],
    'Frequência_Absoluta': [
        len(df_diabeticos),
        len(df_saudaveis),
        len(df)
    ]
})

freq_diabetes['Frequência_Relativa (%)'] = (freq_diabetes['Frequência_Absoluta'] / len(df)) * 100
freq_diabetes['Frequência_Relativa_Decimal'] = freq_diabetes['Frequência_Absoluta'] / len(df)

print(freq_diabetes.to_string(index=False))
print("\n")


print("="*80)
print("TABELA DE FREQUÊNCIA POR ESTADO")
print("="*80)

estado_freq = pd.DataFrame({
    'Freq_Abs_Total': df.groupby('Estado').size(),
    'Freq_Abs_Diabeticos': df_diabeticos.groupby('Estado').size(),
    'Freq_Abs_Saudaveis': df_saudaveis.groupby('Estado').size()
}).fillna(0).astype(int)

estado_freq['Freq_Rel_Total (%)'] = (estado_freq['Freq_Abs_Total'] / len(df)) * 100
estado_freq['Freq_Rel_Diabeticos (%)'] = (estado_freq['Freq_Abs_Diabeticos'] / len(df)) * 100
estado_freq['Freq_Rel_Saudaveis (%)'] = (estado_freq['Freq_Abs_Saudaveis'] / len(df)) * 100

estado_freq['Perc_Diabeticos_no_Estado (%)'] = (estado_freq['Freq_Abs_Diabeticos'] / estado_freq['Freq_Abs_Total']) * 100
estado_freq['Perc_Saudaveis_no_Estado (%)'] = (estado_freq['Freq_Abs_Saudaveis'] / estado_freq['Freq_Abs_Total']) * 100

estado_freq_sorted = estado_freq.sort_values('Perc_Diabeticos_no_Estado (%)', ascending=False)

print(estado_freq_sorted.round(2))
print("\n")

estado_freq_sorted.to_csv('../data/processed/frequencias_por_estado.csv')
print("✓ Tabela exportada para: ../data/processed/frequencias_por_estado.csv\n")


print("="*80)
print("TABELA DE FREQUÊNCIA POR REGIÃO")
print("="*80)

regiao_freq = pd.DataFrame({
    'Freq_Abs_Total': df.groupby('Regiao').size(),
    'Freq_Abs_Diabeticos': df_diabeticos.groupby('Regiao').size(),
    'Freq_Abs_Saudaveis': df_saudaveis.groupby('Regiao').size()
}).fillna(0).astype(int)

regiao_freq['Freq_Rel_Total (%)'] = (regiao_freq['Freq_Abs_Total'] / len(df)) * 100
regiao_freq['Freq_Rel_Diabeticos (%)'] = (regiao_freq['Freq_Abs_Diabeticos'] / len(df)) * 100
regiao_freq['Freq_Rel_Saudaveis (%)'] = (regiao_freq['Freq_Abs_Saudaveis'] / len(df)) * 100

regiao_freq['Perc_Diabeticos_na_Regiao (%)'] = (regiao_freq['Freq_Abs_Diabeticos'] / regiao_freq['Freq_Abs_Total']) * 100
regiao_freq['Perc_Saudaveis_na_Regiao (%)'] = (regiao_freq['Freq_Abs_Saudaveis'] / regiao_freq['Freq_Abs_Total']) * 100

regiao_freq_sorted = regiao_freq.sort_values('Perc_Diabeticos_na_Regiao (%)', ascending=False)

print(regiao_freq_sorted.round(2))
print("\n")

regiao_freq_sorted.to_csv('../data/processed/frequencias_por_regiao.csv')
print("✓ Tabela exportada para: ../data/processed/frequencias_por_regiao.csv\n")


fig, ax = plt.subplots(figsize=(14, 8))
estado_freq_sorted['Freq_Abs_Diabeticos'].plot(kind='barh', ax=ax, color='coral', edgecolor='black')
ax.set_xlabel('Frequência Absoluta (Nº de Diabéticos)', fontsize=12, fontweight='bold')
ax.set_ylabel('Estado', fontsize=12, fontweight='bold')
ax.set_title('Frequência Absoluta de Diabéticos por Estado', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../data/processed/freq_abs_diabeticos_estado.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(14, 8))
estado_freq_sorted['Freq_Rel_Diabeticos (%)'].plot(kind='barh', ax=ax, color='lightcoral', edgecolor='black')
ax.set_xlabel('Frequência Relativa (% do Total Nacional)', fontsize=12, fontweight='bold')
ax.set_ylabel('Estado', fontsize=12, fontweight='bold')
ax.set_title('Frequência Relativa de Diabéticos por Estado\n(em relação ao total nacional)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../data/processed/freq_rel_diabeticos_estado.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(regiao_freq_sorted))
width = 0.35

bars1 = ax.bar(x - width/2, regiao_freq_sorted['Freq_Abs_Diabeticos'], width, 
               label='Diabéticos', color='coral', edgecolor='black')
bars2 = ax.bar(x + width/2, regiao_freq_sorted['Freq_Abs_Saudaveis'], width, 
               label='Saudáveis', color='lightgreen', edgecolor='black')

ax.set_xlabel('Região', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequência Absoluta (Nº de Pessoas)', fontsize=12, fontweight='bold')
ax.set_title('Frequência Absoluta: Diabéticos vs Saudáveis por Região', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regiao_freq_sorted.index, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../data/processed/freq_abs_regiao_comparacao.png', dpi=300, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

wedges1, texts1, autotexts1 = ax1.pie(
    regiao_freq_sorted['Freq_Abs_Total'], 
    labels=regiao_freq_sorted.index,
    autopct='%1.1f%%',
    colors=colors1,
    startangle=90,
    explode=[0.05] * len(regiao_freq_sorted)
)
ax1.set_title('Frequência Relativa: Total de Entrevistados por Região', 
              fontsize=12, fontweight='bold', pad=20)

wedges2, texts2, autotexts2 = ax2.pie(
    regiao_freq_sorted['Freq_Abs_Diabeticos'], 
    labels=regiao_freq_sorted.index,
    autopct='%1.1f%%',
    colors=colors1,
    startangle=90,
    explode=[0.05] * len(regiao_freq_sorted)
)
ax2.set_title('Frequência Relativa: Diabéticos por Região', 
              fontsize=12, fontweight='bold', pad=20)

for autotext in autotexts1 + autotexts2:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

plt.tight_layout()
plt.savefig('../data/processed/freq_rel_regiao_pizza.png', dpi=300, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(regiao_freq_sorted.index, regiao_freq_sorted['Perc_Diabeticos_na_Regiao (%)'], 
              color=colors_bar, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Região', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentual de Diabéticos na Região (%)', fontsize=12, fontweight='bold')
ax.set_title('Percentual de Diabéticos em Relação aos Entrevistados da Região', 
             fontsize=14, fontweight='bold')
ax.set_xticklabels(regiao_freq_sorted.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('../data/processed/perc_diabeticos_regiao.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n" + "="*80)
print("RESUMO CONSOLIDADO - FREQUÊNCIAS")
print("="*80)

resumo = pd.DataFrame({
    'Métrica': [
        'Total de Entrevistados',
        'Total de Diabéticos',
        'Total de Saudáveis',
        'Freq. Relativa Diabéticos (%)',
        'Freq. Relativa Saudáveis (%)',
        '',
        'Estados Analisados',
        'Regiões Analisadas',
        '',
        'Estado com Maior % Diabetes',
        'Estado com Menor % Diabetes',
        '',
        'Região com Maior % Diabetes',
        'Região com Menor % Diabetes'
    ],
    'Valor': [
        f"{len(df):,}",
        f"{len(df_diabeticos):,}",
        f"{len(df_saudaveis):,}",
        f"{(len(df_diabeticos)/len(df))*100:.2f}%",
        f"{(len(df_saudaveis)/len(df))*100:.2f}%",
        '',
        len(estado_freq),
        len(regiao_freq),
        '',
        f"{estado_freq_sorted.index[0]} ({estado_freq_sorted['Perc_Diabeticos_no_Estado (%)'].iloc[0]:.2f}%)",
        f"{estado_freq_sorted.index[-1]} ({estado_freq_sorted['Perc_Diabeticos_no_Estado (%)'].iloc[-1]:.2f}%)",
        '',
        f"{regiao_freq_sorted.index[0]} ({regiao_freq_sorted['Perc_Diabeticos_na_Regiao (%)'].iloc[0]:.2f}%)",
        f"{regiao_freq_sorted.index[-1]} ({regiao_freq_sorted['Perc_Diabeticos_na_Regiao (%)'].iloc[-1]:.2f}%)"
    ]
})

print(resumo.to_string(index=False))
print("\n")

resumo.to_csv('../data/processed/resumo_frequencias.csv', index=False)
print("✓ Resumo exportado para: ../data/processed/resumo_frequencias.csv")

print("\n" + "="*80)
print("✓ ANÁLISE COMPLETA FINALIZADA!")
print("✓ Arquivos CSV gerados:")
print("  - estatisticas_por_estado.csv")
print("  - estatisticas_por_regiao.csv")
print("  - frequencias_por_estado.csv")
print("  - frequencias_por_regiao.csv")
print("  - resumo_frequencias.csv")
print("✓ Gráficos em PNG (alta resolução) gerados:")
print("  - freq_abs_diabeticos_estado.png")
print("  - freq_rel_diabeticos_estado.png")
print("  - freq_abs_regiao_comparacao.png")
print("  - freq_rel_regiao_pizza.png")
print("  - perc_diabeticos_regiao.png")
print("="*80)

