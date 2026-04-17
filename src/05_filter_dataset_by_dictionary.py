import json
from pathlib import Path

import pandas as pd


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]

	input_csv_path = project_root / "data" / "processed" / "01_pns2019_filtered_by_age_and_diabetes.csv"
	dictionary_path = project_root / "data" / "raw" / "pns_dictionary_and_form" / "dictionary" / "filtered_atributes_dictionary.json"
	output_csv_path = project_root / "data" / "processed" / "02_pns2019_filtered_by_dictionary.csv"

	with dictionary_path.open("r", encoding="utf-8") as file:
		filtered_dictionary = json.load(file)

	selected_columns = list(filtered_dictionary.keys())

	df = pd.read_csv(input_csv_path, low_memory=False)

	existing_columns = [column for column in selected_columns if column in df.columns]
	missing_columns = [column for column in selected_columns if column not in df.columns]

	if not existing_columns:
		raise ValueError("Nenhuma coluna do dicionario foi encontrada no CSV.")

	filtered_df = df.loc[:, existing_columns]
	filtered_df.to_csv(output_csv_path, index=False)

	print(f"Arquivo de entrada: {input_csv_path}")
	print(f"Dicionario usado: {dictionary_path}")
	print(f"Arquivo gerado: {output_csv_path}")
	print(f"Total de colunas no CSV de entrada: {len(df.columns)}")
	print(f"Total de colunas solicitadas no dicionario: {len(selected_columns)}")
	print(f"Total de colunas exportadas: {len(existing_columns)}")
	print(f"Total de colunas nao encontradas: {len(missing_columns)}")
	print(f"Total de linhas mantidas: {len(filtered_df)}")

	if missing_columns:
		print("\nColunas do dicionario que nao estao no CSV:")
		for column in missing_columns:
			print(f"- {column}")


if __name__ == "__main__":
	main()
