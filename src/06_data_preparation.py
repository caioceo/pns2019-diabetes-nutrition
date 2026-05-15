import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    input_path = os.path.join(BASE_DIR, "data", "processed", "05_pns2019_clean.csv")
    output_path = os.path.join(BASE_DIR, "data", "processed", "06_pns2019_prepared.csv")

    print("[INFO] Carregando a base limpa...")
    df = pd.read_csv(input_path)

    # 1. Discretização de Idade (C008)
    print("[INFO] Criando categorias de Idade...")
    bins_idade = [29, 39, 49, 61]
    labels_idade = ['Adulto Jovem (30-39)', 'Adulto (40-49)', 'Meia-Idade (50-60)']
    df['C008_Categoria'] = pd.cut(df['C008'], bins=bins_idade, labels=labels_idade)

    # 2. Combinação e Discretização do IMC (P00103 e P00403)
    print("[INFO] Calculando IMC e categorias...")
    # P00403 está em cm, converter para metros
    altura_m = df['P00403'] / 100
    df['Calculo_IMC'] = df['P00103'] / (altura_m ** 2)
    
    bins_imc = [0, 18.49, 24.99, 29.99, 100]
    labels_imc = ['Baixo Peso', 'Eutrófico (Adequado)', 'Sobrepeso', 'Obeso']
    df['IMC_Categoria'] = pd.cut(df['Calculo_IMC'], bins=bins_imc, labels=labels_imc)

    # 3. Fusão: Consumo de Ultraprocessados (P00614 a P00623)
    print("[INFO] Calculando Score de Ultraprocessados...")
    # Colunas de consumo de ontem (1 = Sim, 2 = Não). Vamos converter 1->1, 2->0
    cols_ultra = [f'P006{i:02d}' for i in range(14, 24)] # 14 até 23
    
    # Criar DataFrame temporário mapeado
    temp_ultra = df[cols_ultra].replace({2: 0, 1: 1})
    df['Score_Ultraprocessados_Ontem'] = temp_ultra.sum(axis=1)

    # 4. Fusão: Score de Saúde Mental (N010 a N017)
    print("[INFO] Calculando Score de Saúde Mental...")
    cols_mental = [f'N01{i}' for i in range(0, 8)]
    # Valores originais: 1, 2, 3, 4. Vamos mapear para 0, 1, 2, 3 (padrão PHQ-9)
    temp_mental = df[cols_mental] - 1
    df['Score_Saude_Mental'] = temp_mental.sum(axis=1)

    # 5. Combinação: Nível de Atividade Física OMS
    print("[INFO] Classificando Nível de Atividade Física...")
    # Preencher NaNs com 0 para o cálculo (sedentários estruturais)
    dias_semana = df['P035'].fillna(0)
    horas_dia = df['P03701'].fillna(0)
    min_dia = df['P03702'].fillna(0)
    
    tempo_semanal = dias_semana * ((horas_dia * 60) + min_dia)
    df['Minutos_Semanais_Exercicio'] = tempo_semanal

    def classificar_sedentarismo(row):
        if row['P034'] == 2:
            return 'Sedentário Absoluto'
        elif row['Minutos_Semanais_Exercicio'] >= 150:
            return 'Ativo (>= 150min)'
        else:
            return 'Insuficientemente Ativo (< 150min)'
            
    df['Nivel_Atividade_Fisica'] = df.apply(classificar_sedentarismo, axis=1)

    # 6. Limpeza das Colunas Originais
    print("[INFO] Removendo as colunas originais (evitando multicolinearidade)...")
    cols_to_drop = [
        'C008', # Idade original
        'P00103', 'P00104', 'P00403', 'P00404', # Pesos e Alturas brutos
        'P034', 'P035', 'P03701', 'P03702', # Módulo de exercícios brutos
    ] + cols_ultra + cols_mental # As 10 do consumo de ontem e as 8 de saúde mental

    # Drop seguro ignorando erros caso a coluna já não exista
    df = df.drop(columns=cols_to_drop, errors='ignore')


    print("[INFO] Salvando a base preparada...")
    df.to_csv(output_path, index=False)
    print(f"[SUCESSO] Engenharia de Variáveis concluída: {output_path}")
    print(f"[RESUMO] Novas colunas geradas: C008_Categoria, Calculo_IMC, IMC_Categoria, Score_Ultraprocessados_Ontem, Score_Saude_Mental, Nivel_Atividade_Fisica")

if __name__ == "__main__":
    main()
