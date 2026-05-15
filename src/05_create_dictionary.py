import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Caminhos
    json_path = os.path.join(BASE_DIR, "data", "original_dictionary.json")
    csv_path = os.path.join(BASE_DIR, "data", "processed", "05_pns2019_clean.csv")
    out_json_path = os.path.join(BASE_DIR, "data", "processed", "05_pns2019_clean_dictionary.json")

    print("[INFO] Carregando dicionário original...")
    with open(json_path, "r", encoding="utf-8") as f:
        orig_dict = json.load(f)

    print("[INFO] Lendo cabeçalho da base limpa...")
    df = pd.read_csv(csv_path, nrows=0) # Ler só o cabeçalho
    clean_cols = df.columns.tolist()

    new_dict = {}

    for col in clean_cols:
        if col in orig_dict:
            new_dict[col] = orig_dict[col]
        elif col.endswith("_outlier_flag"):
            base_col = col.replace("_outlier_flag", "")
            if base_col in orig_dict:
                new_dict[col] = "[FLAG OUTLIER] " + orig_dict[base_col]
            else:
                new_dict[col] = "[FLAG OUTLIER] Variável clínica com valor extremo"
        else:
            new_dict[col] = "Descrição não encontrada no dicionário original"

    print("[INFO] Salvando novo dicionário mapeado...")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)

    print(f"[SUCESSO] Dicionário criado: {out_json_path}")
    print(f"[RESUMO] Total de atributos na base limpa: {len(new_dict)}")

    # -- Nova parte: Dicionário de Não Coletados --
    full_dict_path = os.path.join(BASE_DIR, "data", "processed_original_dictionary.json")
    out_unused_path = os.path.join(BASE_DIR, "data", "processed", "05_pns2019_unused_dictionary.json")

    print("[INFO] Carregando dicionário original completo (1087 atributos)...")
    with open(full_dict_path, "r", encoding="utf-8") as f:
        full_dict = json.load(f)
    
    # Remover do dicionário completo tudo o que está no dicionário limpo
    unused_dict = {k: v for k, v in full_dict.items() if k not in new_dict}
    
    # Remover a coluna target e chaves se quisermos? Não, o target 'Q03001' deve estar no new_dict. 
    # Flags de outlier criadas também não estarão no full_dict, o que é perfeito.

    print("[INFO] Salvando dicionário de atributos descartados/não coletados...")
    with open(out_unused_path, "w", encoding="utf-8") as f:
        json.dump(unused_dict, f, ensure_ascii=False, indent=4)

    print(f"[SUCESSO] Dicionário de não coletados criado: {out_unused_path}")
    print(f"[RESUMO] Total de atributos descartados: {len(unused_dict)}")

if __name__ == "__main__":
    main()
