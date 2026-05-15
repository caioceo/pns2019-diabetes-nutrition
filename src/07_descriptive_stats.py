import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    path_06 = os.path.join(BASE_DIR, "data", "processed", "06_pns2019_prepared.csv")
    
    print("[INFO] Lendo a base final preparada...")
    df = pd.read_csv(path_06)
    
    total = len(df)
    print(f"Total de registros: {total}")
    
    # 1. Sexo (C006: 1 = Homem, 2 = Mulher)
    if 'C006' in df.columns:
        sexo_counts = df['C006'].value_counts(normalize=True) * 100
        print("\n--- SEXO ---")
        print(f"Homens (1): {sexo_counts.get(1.0, 0):.2f}%")
        print(f"Mulheres (2): {sexo_counts.get(2.0, 0):.2f}%")
        
    # 2. Idade Categoria
    if 'C008_Categoria' in df.columns:
        idade_counts = df['C008_Categoria'].value_counts(normalize=True) * 100
        print("\n--- FAIXA ETÁRIA ---")
        print(idade_counts.to_string())
        
    # 3. Regionalidade (V0001 - UF)
    # Regiões: Norte (11-17), Nordeste (21-29), Sudeste (31-35), Sul (41-43), Centro-Oeste (50-53)
    if 'V0001' in df.columns:
        def get_regiao(uf):
            uf = int(uf)
            if 11 <= uf <= 17: return 'Norte'
            if 21 <= uf <= 29: return 'Nordeste'
            if 31 <= uf <= 35: return 'Sudeste'
            if 41 <= uf <= 43: return 'Sul'
            if 50 <= uf <= 53: return 'Centro-Oeste'
            return 'Desconhecido'
            
        df['Regiao'] = df['V0001'].apply(get_regiao)
        regiao_counts = df['Regiao'].value_counts(normalize=True) * 100
        print("\n--- REGIÕES DO BRASIL ---")
        print(regiao_counts.to_string())

    # 4. IMC
    if 'IMC_Categoria' in df.columns:
        imc_counts = df['IMC_Categoria'].value_counts(normalize=True) * 100
        print("\n--- ESTADO NUTRICIONAL (IMC) ---")
        print(imc_counts.to_string())
        
    # 5. Diabetes (Q03001: 1=Sim, 2/3=Não)
    if 'Q03001' in df.columns:
        diabetes_counts = df['Q03001'].value_counts()
        print("\n--- PREVALÊNCIA DE DIABETES ---")
        print(diabetes_counts.to_string())

if __name__ == "__main__":
    main()
