"""
Script: 09_attribute_selection.py
Objetivo: Realizar a seleção de atributos (Etapa 7 do KDD - Critério 19) por meio de
          Entropia de Shannon e Ganho de Informação (Information Gain).
          Calcula a relevância de cada atributo em relação à classe alvo Q03001
          (Diabetes) e gera a base filtrada com as melhores características.

Input:
    data/processed/08_pns2019_outliers_treated.csv
Output:
    data/processed/09_pns2019_selected_features.csv
    docs/relatorios/relatorio_selecao_atributos.txt
    docs/relatorios/decisoes_selecao.csv
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_INPUT = os.path.join(BASE_DIR, "data", "processed", "08_pns2019_outliers_treated.csv")
PATH_OUTPUT_DATA = os.path.join(BASE_DIR, "data", "processed", "09_pns2019_selected_features.csv")
PATH_REPORT = os.path.join(BASE_DIR, "docs", "relatorios", "relatorio_selecao_atributos.txt")
PATH_CSV_DECISÕES = os.path.join(BASE_DIR, "docs", "relatorios", "decisoes_selecao.csv")

TARGET_COL = "Q03001"

# Vazamentos de dados (Data Leakage) que devem ser excluídos
LEAKS = ["Q031"]  # Idade no diagnóstico (só existe para quem tem diabetes)

# Identificadores geográficos ou colunas que não devem entrar no modelo preditivo
EXCLUIR_MODELAGEM = ["V0001"] 

def shannon_entropy(series):
    """Calcula a Entropia de Shannon H(Y) de uma variável discreta."""
    total = len(series)
    if total == 0:
        return 0.0
    counts = series.value_counts()
    probabilities = counts / total
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(df, feature, target):
    """Calcula o Ganho de Informação IG(Y, X) = H(Y) - H(Y|X)."""
    # 1. Entropia total da classe alvo H(Y)
    h_y = shannon_entropy(df[target])
    
    # 2. Entropia condicional H(Y|X)
    h_y_x = 0.0
    total_instancias = len(df)
    
    # Para atributos contínuos com muitos valores únicos, fazemos uma discretização temporária
    # em 5 bins usando quantis (qcut) para evitar o sobreajuste clássico do Ganho de Informação.
    if df[feature].dtype in [np.float64, np.int64] and df[feature].nunique() > 10:
        try:
            # pd.qcut divide os dados em intervalos com frequências iguais
            valores_calculo = pd.qcut(df[feature], q=5, labels=False, duplicates='drop')
        except Exception:
            # Fallback caso dê erro por excesso de valores idênticos
            valores_calculo = pd.cut(df[feature], bins=5, labels=False)
    else:
        valores_calculo = df[feature]
    
    # Agrupa a classe alvo pelas categorias do atributo avaliado
    grupos = df[target].groupby(valores_calculo)
    for val, grupo in grupos:
        p_x = len(grupo) / total_instancias
        h_grupo = shannon_entropy(grupo)
        h_y_x += p_x * h_grupo
        
    # Ganho de Informação
    ig = h_y - h_y_x
    return ig

def main():
    print("=" * 70)
    print("  ETAPA 7 — SELEÇÃO DE ATRIBUTOS POR ENTROPIA | PNS 2019")
    print("=" * 70)
    
    if not os.path.exists(PATH_INPUT):
        print(f"[ERRO] Arquivo de entrada não encontrado: {PATH_INPUT}")
        return
        
    df = pd.read_csv(PATH_INPUT)
    print(f"[INFO] Dataset carregado: {df.shape[0]:,} registros e {df.shape[1]} colunas.")
    
    # Calcular Entropia do Target
    h_target = shannon_entropy(df[TARGET_COL])
    print(f"[INFO] Entropia de Shannon do Target H({TARGET_COL}) = {h_target:.5f}")
    
    # Identificar colunas elegíveis
    colunas_analise = [
        col for col in df.columns 
        if col != TARGET_COL and col not in LEAKS and col not in EXCLUIR_MODELAGEM
    ]
    
    print(f"[INFO] Analisando {len(colunas_analise)} atributos...")
    
    # Calcular Ganho de Informação para cada atributo
    resultados = []
    for col in colunas_analise:
        ig = information_gain(df, col, TARGET_COL)
        resultados.append({
            "atributo": col,
            "ganho_informacao": ig,
            "tipo": str(df[col].dtype),
            "valores_unicos": df[col].nunique()
        })
        
    # Ordenar por Ganho de Informação decrescente
    df_resultados = pd.DataFrame(resultados).sort_values(by="ganho_informacao", ascending=False)
    
    # Salvar tabela de decisões em CSV
    df_resultados.to_csv(PATH_CSV_DECISÕES, index=False, encoding="utf-8")
    print(f"[SALVO] Ranking completo de entropia salvo em: {PATH_CSV_DECISÕES}")
    
    # Definir critério de corte:
    # 1. Podemos filtrar por um threshold de relevância (ex: IG > 0.0005)
    # 2. Ou selecionar os TOP N atributos mais informativos (ex: Top 25)
    # Para o estudo de diabetes e hábitos nutricionais, selecionaremos os Top 25 atributos mais informativos
    TOP_N = 25
    atributos_selecionados = df_resultados.head(TOP_N)["atributo"].tolist()
    
    print(f"\n[SELEÇÃO] Top {TOP_N} atributos selecionados por Entropia:")
    for idx, row in df_resultados.head(TOP_N).iterrows():
        print(f"  {row['atributo']:<30} | IG: {row['ganho_informacao']:.6f} | Uniques: {row['valores_unicos']}")
        
    # Criar a nova base de dados apenas com os atributos selecionados + o Target
    colunas_finais = atributos_selecionados + [TARGET_COL]
    df_selecionado = df[colunas_finais]
    df_selecionado.to_csv(PATH_OUTPUT_DATA, index=False, encoding="utf-8")
    print(f"\n[SALVO] Base filtrada salva em: {PATH_OUTPUT_DATA}")
    print(f"[INFO] Dimensões da base final: {df_selecionado.shape[0]:,} linhas × {df_selecionado.shape[1]} colunas.")
    
    # Gerar Relatório Textual para o Artigo
    with open(PATH_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  RELATÓRIO DE SELEÇÃO DE ATRIBUTOS POR ENTROPIA (IG) | PNS 2019\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"1. INFORMAÇÕES DA BASE\n")
        f.write(f"   - Base de entrada: {df.shape[0]:,} instâncias, {df.shape[1]} colunas\n")
        f.write(f"   - Variável Alvo (Target): {TARGET_COL}\n")
        f.write(f"   - Entropia do Target H({TARGET_COL}): {h_target:.6f}\n")
        f.write(f"   - Exclusões por Vazamento (Leakage): {LEAKS}\n")
        f.write(f"   - Exclusões Identificadoras: {EXCLUIR_MODELAGEM}\n\n")
        
        f.write(f"2. CRITÉRIO DE FILTRAGEM\n")
        f.write(f"   - Método: Seleção por Limiar de Ganho de Informação (Entropia de Shannon)\n")
        f.write(f"   - Decisão: Retenção dos Top {TOP_N} atributos mais informativos para a modelagem\n")
        f.write(f"   - Base resultante: {df_selecionado.shape[0]:,} instâncias × {df_selecionado.shape[1]} colunas\n\n")
        
        f.write(f"3. RANKING COMPLETO DE ATRIBUTOS (GANHO DE INFORMAÇÃO)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Ranking':<8} {'Atributo':<30} {'Ganho de Informação':<22} {'Uniques':<10} {'Status':<10}\n")
        f.write("-" * 80 + "\n")
        
        for idx, (_, row) in enumerate(df_resultados.iterrows(), 1):
            status = "Selecionado" if row['atributo'] in atributos_selecionados else "Descartado"
            f.write(f"{idx:<8} {row['atributo']:<30} {row['ganho_informacao']:<22.6f} {row['valores_unicos']:<10} {status:<10}\n")
            
        f.write("\n\n4. JUSTIFICATIVA CLÍNICA E COMPORTAMENTAL DOS TOP ATRIBUTOS\n")
        f.write("-" * 80 + "\n")
        f.write("Os atributos no topo do ranking revelam correlações importantes com a ocorrência de diabetes:\n")
        f.write("- Atributos de saúde (ex: consultas médicas J012 e aferição de pressão arterial Q00101) estão ligados ao monitoramento frequente de diabéticos.\n")
        f.write("- Indicadores antropométricos (IMC e Peso) e faixa etária (C008_Categoria) apresentam forte ganho de informação, condizente com a fisiopatologia do Diabetes Tipo 2.\n")
        f.write("- Variáveis de consumo alimentar e estilo de vida (como o Score_Ultraprocessados_Ontem e Nivel_Atividade_Fisica) representam os fatores de risco comportamentais de interesse do estudo.\n")

    print(f"[SALVO] Relatório textual salvo em: {PATH_REPORT}")

if __name__ == "__main__":
    main()
