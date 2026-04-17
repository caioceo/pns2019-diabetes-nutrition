# %%

import json
from collections import defaultdict
from pathlib import Path

def normalize_text(text: str) -> str:
	return " ".join(str(text).strip().lower().split())


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	dictionary_dir = project_root / "data" / "raw" / "pns_dictionary_and_form" / "dictionary"

	processed_original_path = dictionary_dir / "processed_original_dictionary.json"
	filtered_attributes_path = dictionary_dir / "filtered_atributes_names.json"
	output_path = dictionary_dir / "filtered_atributes_dictionary.json"

	with processed_original_path.open("r", encoding="utf-8") as file:
		processed_original = json.load(file)

	with filtered_attributes_path.open("r", encoding="utf-8") as file:
		filtered_attributes = json.load(file)

	description_to_codes = defaultdict(list)
	for code, description in processed_original.items():
		description_to_codes[normalize_text(description)].append(code)

	joined_attributes = {}
	not_found = []
	ambiguous = {}

	for description in filtered_attributes:
		key = normalize_text(description)
		matched_codes = description_to_codes.get(key, [])

		if not matched_codes:
			not_found.append(description)
			continue

		if len(matched_codes) > 1:
			ambiguous[description] = matched_codes

		for code in matched_codes:
			joined_attributes[code] = processed_original[code]

	with output_path.open("w", encoding="utf-8") as file:
		json.dump(joined_attributes, file, ensure_ascii=False, indent=2)

	print(f"Arquivo gerado: {output_path}")
	print(f"Atributos filtrados informados: {len(filtered_attributes)}")
	print(f"Atributos encontrados: {len(joined_attributes)}")
	print(f"Nao encontrados: {len(not_found)}")
	print(f"Descricoes ambiguas: {len(ambiguous)}")

	if not_found:
		print("\nDescricoes nao encontradas:")
		for description in not_found:
			print(f"- {description}")

	if ambiguous:
		print("\nDescricoes ambiguas (mesma descricao com multiplos codigos):")
		for description, codes in ambiguous.items():
			print(f"- {description}: {', '.join(codes)}")


if __name__ == "__main__":
	main()
