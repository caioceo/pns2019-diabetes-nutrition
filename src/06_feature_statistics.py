import json
from collections import Counter
from pathlib import Path
import unicodedata

import pandas as pd


NA_TEXT = "Não se aplica"


DIMENSION_CONFIG = {
	"D1": {
		"nome": "Hábitos Alimentares",
		"aspectos": [
			{
				"id": "A1",
				"nome": "Recordatório 24h — consumo ontem",
				"artigos": ["Não informado"],
				"atributos": [
					"P00601",
					"P00602",
					"P00603",
					"P00604",
					"P00605",
					"P00607",
					"P00608",
					"P00609",
					"P00610",
					"P00611",
					"P00613",
					"P00617",
					"P00618",
					"P00619",
					"P00620",
					"P00621",
					"P00622",
					"P00623",
				],
			},
			{
				"id": "A2",
				"nome": "Frequência semanal de grupos alimentares",
				"artigos": ["Não informado"],
				"atributos": [
					"P006",
					"P00901",
					"P01001",
					"P01101",
					"P013",
					"P015",
					"P018",
					"P019",
					"P02501",
					"P02602",
				],
			},
			{
				"id": "A3",
				"nome": "Percepção da qualidade da dieta",
				"artigos": ["Não informado"],
				"atributos": ["P02601"],
			},
		],
	},
	"D2": {
		"nome": "Ingestão de Bebidas e Álcool",
		"aspectos": [
			{
				"id": "B1",
				"nome": "Bebidas açucaradas e leite",
				"artigos": ["Não informado"],
				"atributos": [
					"P01601",
					"P00612",
					"P02002",
					"P02102",
					"P00614",
					"P00615",
					"P00616",
					"P023",
					"P02401",
					"P02001",
					"P02101",
				],
			},
			{
				"id": "B2",
				"nome": "Consumo de álcool",
				"artigos": ["Não informado"],
				"atributos": ["P027", "P02801", "P029", "P03201"],
			},
		],
	},
	"D3": {
		"nome": "Atividade Física e Sedentarismo",
		"aspectos": [
			{
				"id": "C1",
				"nome": "Exercício programado",
				"artigos": ["Não informado"],
				"atributos": ["P034", "P035", "P03701", "P03702", "P036"],
			},
			{
				"id": "C2",
				"nome": "Atividade habitual e deslocamento ativo",
				"artigos": ["Não informado"],
				"atributos": ["P038", "P039", "P040", "P04001", "P04101", "P04102", "VDM001"],
			},
			{
				"id": "C3",
				"nome": "Comportamento sedentário",
				"artigos": ["Não informado"],
				"atributos": ["P04501", "P04502"],
			},
		],
	},
	"D4": {
		"nome": "Perfil Antropométrico",
		"aspectos": [
			{
				"id": "D1",
				"nome": "Medidas corporais",
				"artigos": ["Não informado"],
				"atributos": [
					"W00101",
					"W00102",
					"W00201",
					"W00202",
					"W00203",
					"P00103",
					"P00104",
					"W00103",
					"P00403",
					"P00404",
				],
			}
		],
	},
	"D5": {
		"nome": "Perfil Sociodemográfico e Econômico",
		"aspectos": [
			{
				"id": "E1",
				"nome": "Características pessoais",
				"artigos": ["Não informado"],
				"atributos": ["C006", "C008", "C009", "C011"],
			},
			{
				"id": "E2",
				"nome": "Escolaridade",
				"artigos": ["Não informado"],
				"atributos": ["VDD004A"],
			},
			{
				"id": "E3",
				"nome": "Território, renda e trabalho",
				"artigos": ["Não informado"],
				"atributos": ["V0026", "V0031", "V0001", "VDF003", "VDF004", "VDE014", "E017", "M005010"],
			},
		],
	},
	"D6": {
		"nome": "Saúde e Comportamentos de Risco",
		"aspectos": [
			{
				"id": "F1",
				"nome": "Diagnósticos e acompanhamento clínico",
				"artigos": ["Não informado"],
				"atributos": [
					"Q00201",
					"Q00202",
					"Q00401",
					"Q00503",
					"Q00601",
					"Q00101",
					"Q02901",
					"Q03001",
					"Q031",
					"Q03201",
					"I00102",
					"J012",
				],
			},
			{
				"id": "F2",
				"nome": "Saúde mental",
				"artigos": ["Não informado"],
				"atributos": ["N010", "N011", "N012", "N013", "N014", "N015", "N016", "N017"],
			},
			{
				"id": "F3",
				"nome": "Tabagismo",
				"artigos": ["Não informado"],
				"atributos": ["P050", "P051", "P05401", "P060"],
			},
		],
	},
}


NOMINAL_CODES = {
	"C006",
	"C009",
	"C011",
	"V0026",
	"V0031",
	"V0001",
	"VDD004A",
	"P01001",
	"P02102",
	"P02101",
	"P02401",
	"P036",
	"I00102",
	"VDE014",
}


ARTICLE_REFERENCES = {
	"CAPTO": "CAPTO - A Method for Understanding Problem Domains for Data Science Projects (Zárate et al.)",
	"MALTA_HBA1C": "Validade do diabetes autorreferido e a relação com a hemoglobina glicada na população brasileira (Malta et al.)",
	"ADA": "Nutrition Principles and Recommendations in Diabetes (American Diabetes Association)",
	"CLARO_LOUZADA": "Consumo de alimentos não saudáveis (Claro et al.) + Alimentos ultraprocessados e perfil nutricional (Louzada et al.)",
	"MALTA_SOCIO": "Socioeconomic inequalities related to noncommunicable diseases and their limitations (Malta et al.)",
	"SOUZA_SOCIO": "Desigualdades sociodemográficas no consumo alimentar no Brasil (Souza et al.)",
	"NUNES_MULTIMORB": "Desigualdades sociodemográficas na ocorrência de multi-morbidade no Brasil (Nunes et al.)",
	"LOPES_DEPRESSAO": "Depressão no Brasil: resultados da Pesquisa Nacional de Saúde (Lopes et al.)",
	"KAVAKIOTIS_ML": "Machine Learning and Data Mining Methods in Diabetes Research (Kavakiotis et al.)",
	"MALTA_AF": "Prática de atividade física e comportamento sedentário (Malta et al.)",
	"GARCIA_AF": "Prática de atividade física no tempo livre e deslocamento ativo (Garcia et al.)",
}


ARTICLE_REASON = {
	"CAPTO": "Método usado para estruturar o domínio e organizar variáveis em dimensões.",
	"MALTA_HBA1C": "Fundamenta a validade do diagnóstico autorreferido de diabetes como variável-alvo.",
	"ADA": "Diretriz clínica que sustenta o impacto de padrões alimentares no controle glicêmico.",
	"CLARO_LOUZADA": "Evidência brasileira sobre o papel de ultraprocessados e bebidas açucaradas no risco crônico.",
	"MALTA_SOCIO": "Mostra a influência das desigualdades socioeconômicas no perfil de doença crônica e acesso em saúde.",
	"SOUZA_SOCIO": "Demonstra diferenças sociodemográficas no padrão de consumo alimentar no Brasil.",
	"NUNES_MULTIMORB": "Sustenta abordagem clínica integrada para multimorbidade e hipertensão associada ao diabetes.",
	"LOPES_DEPRESSAO": "Relaciona sintomas depressivos e sono com condições crônicas de saúde.",
	"KAVAKIOTIS_ML": "Revisão em TI/ML que destaca variáveis clínicas e comportamentais relevantes para predição em diabetes.",
	"MALTA_AF": "Sustenta a inclusão de atividade física e comportamento sedentário no modelo.",
	"GARCIA_AF": "Justifica atividade física no tempo livre e deslocamento ativo como determinantes de risco.",
}


ASPECT_ARTICLE_KEYS = {
	("D1", "A1"): ["ADA", "CLARO_LOUZADA"],
	("D1", "A2"): ["ADA", "CLARO_LOUZADA"],
	("D1", "A3"): ["ADA"],
	("D2", "B1"): ["CLARO_LOUZADA", "ADA"],
	("D2", "B2"): ["KAVAKIOTIS_ML"],
	("D3", "C1"): ["GARCIA_AF", "MALTA_AF"],
	("D3", "C2"): ["GARCIA_AF", "MALTA_AF"],
	("D3", "C3"): ["MALTA_AF"],
	("D4", "D1"): ["KAVAKIOTIS_ML"],
	("D5", "E1"): ["SOUZA_SOCIO", "MALTA_SOCIO"],
	("D5", "E2"): ["SOUZA_SOCIO"],
	("D5", "E3"): ["MALTA_SOCIO", "SOUZA_SOCIO"],
	("D6", "F1"): ["NUNES_MULTIMORB", "MALTA_HBA1C", "KAVAKIOTIS_ML"],
	("D6", "F2"): ["LOPES_DEPRESSAO"],
	("D6", "F3"): ["KAVAKIOTIS_ML"],
}


ASPECT_PRIMARY_ARTICLE_KEY = {
	("D1", "A1"): "ADA",
	("D1", "A2"): "ADA",
	("D1", "A3"): "ADA",
	("D2", "B1"): "CLARO_LOUZADA",
	("D2", "B2"): "KAVAKIOTIS_ML",
	("D3", "C1"): "GARCIA_AF",
	("D3", "C2"): "GARCIA_AF",
	("D3", "C3"): "MALTA_AF",
	("D4", "D1"): "KAVAKIOTIS_ML",
	("D5", "E1"): "SOUZA_SOCIO",
	("D5", "E2"): "SOUZA_SOCIO",
	("D5", "E3"): "MALTA_SOCIO",
	("D6", "F1"): "NUNES_MULTIMORB",
	("D6", "F2"): "LOPES_DEPRESSAO",
	("D6", "F3"): "KAVAKIOTIS_ML",
}


ATTRIBUTE_PRIMARY_ARTICLE_OVERRIDES = {
	"P00617": "CLARO_LOUZADA",
	"P00618": "CLARO_LOUZADA",
	"P00619": "CLARO_LOUZADA",
	"P00620": "CLARO_LOUZADA",
	"P00621": "CLARO_LOUZADA",
	"P00622": "CLARO_LOUZADA",
	"P00623": "CLARO_LOUZADA",
	"P02001": "CLARO_LOUZADA",
	"P02101": "CLARO_LOUZADA",
	"P02102": "CLARO_LOUZADA",
	"Q02901": "KAVAKIOTIS_ML",
	"Q03001": "MALTA_HBA1C",
	"Q03201": "KAVAKIOTIS_ML",
	"I00102": "MALTA_SOCIO",
	"J012": "KAVAKIOTIS_ML",
}


def slugify_text(text):
	normalized = unicodedata.normalize("NFKD", text)
	ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
	safe = "".join(char if char.isalnum() else "_" for char in ascii_text.lower())
	while "__" in safe:
		safe = safe.replace("__", "_")
	return safe.strip("_")


def format_number(value):
	if pd.isna(value):
		return NA_TEXT

	as_float = float(value)
	if as_float.is_integer():
		return str(int(as_float))

	return f"{as_float:.4f}"


def format_mode(series):
	modes = series.mode(dropna=True)
	if modes.empty:
		return NA_TEXT

	mode_values = [str(v) for v in modes.iloc[:5].tolist()]
	if len(modes) > 5:
		mode_values.append("...")

	return " | ".join(mode_values)


def format_frequency(series, max_categories=10):
	counts = series.value_counts(dropna=True)
	if counts.empty:
		return NA_TEXT

	total = int(counts.sum())
	chunks = []

	for value, count in counts.iloc[:max_categories].items():
		pct = (count / total) * 100
		chunks.append(f"{value}: {int(count)} ({pct:.2f}%)")

	if len(counts) > max_categories:
		chunks.append(f"... +{len(counts) - max_categories} categorias")

	return " ; ".join(chunks)


def infer_category(code, description, series):
	if code in NOMINAL_CODES:
		return "Nominal"

	non_null = series.dropna()
	if non_null.empty:
		return "Ordinal"

	unique_count = non_null.nunique(dropna=True)
	if unique_count <= 2:
		return "Nominal"

	description_lower = description.lower()
	nominal_keywords = [
		"que tipo",
		"sexo",
		"cor",
		"raça",
		"estado civil",
		"unidade da federação",
		"grupamentos",
		"qual o exercício",
	]

	if any(keyword in description_lower for keyword in nominal_keywords):
		return "Nominal"

	return "Ordinal"


def calculate_attribute_stats(series, category):
	cleaned = series.dropna()

	mode_value = format_mode(cleaned)
	freq_value = format_frequency(cleaned)

	if category == "Nominal":
		min_value = NA_TEXT
		max_value = NA_TEXT
		mean_value = NA_TEXT
		std_value = NA_TEXT
	else:
		numeric = pd.to_numeric(cleaned, errors="coerce").dropna()

		if numeric.empty:
			min_value = NA_TEXT
			max_value = NA_TEXT
			mean_value = NA_TEXT
			std_value = NA_TEXT
		else:
			min_value = format_number(numeric.min())
			max_value = format_number(numeric.max())
			mean_value = format_number(numeric.mean())
			std_value = format_number(numeric.std())

	info_text = (
		f"Categoria: {category} | Min: {min_value} | Max: {max_value} | "
		f"Média: {mean_value} | Desvio padrão: {std_value} | "
		f"Moda: {mode_value} | Frequência: {freq_value}"
	)

	return {
		"Categoria (nominal ou ordinal)": category,
		"Valor min": min_value,
		"Valor max": max_value,
		"Média": mean_value,
		"Desvio padrão": std_value,
		"Moda": mode_value,
		"Frequência": freq_value,
		"Informações atributos": info_text,
	}


def validate_dimension_config(attribute_codes):
	mapped_codes = []
	for dimension_data in DIMENSION_CONFIG.values():
		for aspect in dimension_data["aspectos"]:
			mapped_codes.extend(aspect["atributos"])

	mapped_counter = Counter(mapped_codes)
	duplicates = sorted([code for code, count in mapped_counter.items() if count > 1])
	missing = sorted(set(attribute_codes) - set(mapped_codes))
	unknown = sorted(set(mapped_codes) - set(attribute_codes))

	errors = []
	if duplicates:
		errors.append(f"Atributos duplicados em mais de um aspecto: {duplicates}")
	if missing:
		errors.append(f"Atributos sem dimensão/aspecto no config: {missing}")
	if unknown:
		errors.append(f"Atributos no config que não existem no JSON: {unknown}")

	if errors:
		raise ValueError("\n".join(errors))


def get_aspect_article_keys(dimension_id, aspect_id):
	return ASPECT_ARTICLE_KEYS.get((dimension_id, aspect_id), ["CAPTO"])


def select_primary_article_key(dimension_id, aspect_id, code):
	if code in ATTRIBUTE_PRIMARY_ARTICLE_OVERRIDES:
		return ATTRIBUTE_PRIMARY_ARTICLE_OVERRIDES[code]

	return ASPECT_PRIMARY_ARTICLE_KEY.get((dimension_id, aspect_id), "CAPTO")


def validate_article_mapping():
	used_keys = set()

	for article_keys in ASPECT_ARTICLE_KEYS.values():
		used_keys.update(article_keys)

	used_keys.update(ASPECT_PRIMARY_ARTICLE_KEY.values())
	used_keys.update(ATTRIBUTE_PRIMARY_ARTICLE_OVERRIDES.values())

	unknown_keys = sorted(key for key in used_keys if key not in ARTICLE_REFERENCES)
	missing_reason = sorted(key for key in used_keys if key not in ARTICLE_REASON)

	errors = []
	if unknown_keys:
		errors.append(f"Chaves de artigo sem referência em ARTICLE_REFERENCES: {unknown_keys}")
	if missing_reason:
		errors.append(f"Chaves de artigo sem justificativa em ARTICLE_REASON: {missing_reason}")

	if errors:
		raise ValueError("\n".join(errors))


def build_dimension_tables(df, attributes_dict):
	all_dimension_tables = {}
	consolidated_rows = []

	for dimension_id, dimension_data in DIMENSION_CONFIG.items():
		dimension_rows = []
		dimension_name = dimension_data["nome"]

		for aspect in dimension_data["aspectos"]:
			aspect_id = aspect["id"]
			aspect_label = f"{aspect['id']} - {aspect['nome']}"
			aspect_article_keys = get_aspect_article_keys(dimension_id, aspect_id)
			aspect_articles = [ARTICLE_REFERENCES[key] for key in aspect_article_keys]
			articles_text = " ; ".join(aspect_articles)
			aspect_col_text = f"{aspect_label} | Artigos: {articles_text}"

			for code in aspect["atributos"]:
				description = attributes_dict[code]
				linked_attribute = f"{code} - {description}"
				primary_article_key = select_primary_article_key(dimension_id, aspect_id, code)

				if code not in df.columns:
					stats = {
						"Categoria (nominal ou ordinal)": NA_TEXT,
						"Valor min": NA_TEXT,
						"Valor max": NA_TEXT,
						"Média": NA_TEXT,
						"Desvio padrão": NA_TEXT,
						"Moda": NA_TEXT,
						"Frequência": NA_TEXT,
						"Informações atributos": "Atributo não encontrado no dataframe.",
					}
				else:
					series = df[code]
					category = infer_category(code, description, series)
					stats = calculate_attribute_stats(series, category)

				row = {
					"Dimensão": dimension_id,
					"Título da dimensão": dimension_name,
					"Aspectos (conhecimento explicito e estudo cientifico vinculado)": aspect_col_text,
					"Atributos vinculados": linked_attribute,
					"Artigo principal de justificativa": ARTICLE_REFERENCES[primary_article_key],
					"Motivo da justificativa principal": ARTICLE_REASON[primary_article_key],
					**stats,
				}

				dimension_rows.append(row)
				consolidated_rows.append(row)

		all_dimension_tables[dimension_id] = pd.DataFrame(dimension_rows)

	consolidated_df = pd.DataFrame(consolidated_rows)
	column_order = [
		"Dimensão",
		"Título da dimensão",
		"Aspectos (conhecimento explicito e estudo cientifico vinculado)",
		"Atributos vinculados",
		"Artigo principal de justificativa",
		"Motivo da justificativa principal",
		"Categoria (nominal ou ordinal)",
		"Valor min",
		"Valor max",
		"Média",
		"Desvio padrão",
		"Moda",
		"Frequência",
		"Informações atributos",
	]
	consolidated_df = consolidated_df[column_order]
	return all_dimension_tables, consolidated_df


def save_tables(dimension_tables, consolidated_df, output_dir):
	output_dir.mkdir(parents=True, exist_ok=True)

	for legacy_file in output_dir.glob("03_feature_statistics_D*.csv"):
		legacy_file.unlink(missing_ok=True)

	legacy_consolidated = output_dir / "03_feature_statistics_consolidado.csv"
	legacy_consolidated.unlink(missing_ok=True)

	consolidated_path = output_dir / "03_feature_statistics_table.csv"
	consolidated_df.to_csv(consolidated_path, index=False, encoding="utf-8-sig")
	return consolidated_path


def main():
	project_root = Path(__file__).resolve().parents[1]

	data_path = project_root / "data" / "processed" / "02_pns2019_filtered_by_dictionary.csv"
	dictionary_path = project_root / "data" / "raw" / "pns_dictionary_and_form" / "dictionary" / "filtered_atributes_dictionary.json"
	output_dir = project_root / "data" / "processed"

	with dictionary_path.open("r", encoding="utf-8") as file:
		attributes_dict = json.load(file)

	validate_dimension_config(list(attributes_dict.keys()))
	validate_article_mapping()

	df = pd.read_csv(data_path, low_memory=False)

	dimension_tables, consolidated_df = build_dimension_tables(df, attributes_dict)
	single_table_path = save_tables(dimension_tables, consolidated_df, output_dir)

	print("Tabelas geradas com sucesso.")
	print(f"Tabela única: {single_table_path}")

	print("\nContagem de atributos por dimensão:")
	for dimension_id, table in dimension_tables.items():
		print(f"{dimension_id}: {len(table)}")


if __name__ == "__main__":
	main()
