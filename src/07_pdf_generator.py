import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DEFAULT_OUTPUT_NAME = "04_feature_statistics_table_formal.pdf"

DOCUMENT_HEADER_LINES = [
	"Descrição Atributo do Mapa Conceitual",
	"Descrição do Atributos – Domínio de problema: Diabetes",
	"Base de dados: Pessoas Saudáveis e Doentes entre 30 e 60 anos que responderam toda a PNS",
]

DIMENSION_CODE_COLUMN = "Dimensão"
DIMENSION_TITLE_COLUMN = "Título da dimensão"
ASPECT_COLUMN = "Aspectos (conhecimento explicito e estudo cientifico vinculado)"
ATTRIBUTE_COLUMN = "Atributos vinculados"
ARTICLE_SPLIT_MARKER = "| Artigos:"
METRIC_COLUMNS = [
	"Categoria (nominal ou ordinal)",
	"Valor min",
	"Valor max",
	"Média",
	"Desvio padrão",
	"Moda",
	"Frequência",
]


def resolve_pdf_fonts():
	font_dir = Path("C:/Windows/Fonts")
	regular_candidates = ["times.ttf", "Times New Roman.ttf", "TIMES.TTF"]
	bold_candidates = ["timesbd.ttf", "Times New Roman Bold.ttf", "TIMESBD.TTF"]

	def find_first_existing(candidates):
		for candidate in candidates:
			font_path = font_dir / candidate
			if font_path.exists():
				return font_path
		return None

	regular_path = find_first_existing(regular_candidates)
	bold_path = find_first_existing(bold_candidates)

	if regular_path and bold_path:
		regular_name = "TimesNewRomanCustom"
		bold_name = "TimesNewRomanCustom-Bold"

		registered = pdfmetrics.getRegisteredFontNames()
		if regular_name not in registered:
			pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
		if bold_name not in registered:
			pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))

		return regular_name, bold_name

	return "Times-Roman", "Times-Bold"


def resolve_input_path(project_root, input_arg):
	if input_arg:
		candidate = Path(input_arg)
		if not candidate.is_absolute():
			candidate = project_root / candidate
		return candidate

	candidates = [
		project_root / "data" / "processed" / "03_feature_statistics_table.csv",
		project_root / "data" / "processed" / "03_feature_statistics_tabela_unica.csv",
	]

	for candidate in candidates:
		if candidate.exists():
			return candidate

	raise FileNotFoundError("Nenhum CSV de estatisticas encontrado em data/processed.")


def resolve_output_path(project_root, output_arg):
	if output_arg:
		output = Path(output_arg)
		if not output.is_absolute():
			output = project_root / output
		return output

	return project_root / "data" / "processed" / DEFAULT_OUTPUT_NAME


def safe_text(value):
	if pd.isna(value):
		return "Nao se aplica"

	text = str(value).strip()
	if not text:
		return "Nao se aplica"

	return text


def extract_articles_only(aspect_value):
	text = safe_text(aspect_value)
	if ARTICLE_SPLIT_MARKER in text:
		return text.split(ARTICLE_SPLIT_MARKER, 1)[1].strip()

	return text


def order_dataframe_by_dimension(df):
	ordered = df.copy()
	dimension_numbers = pd.to_numeric(
		ordered[DIMENSION_CODE_COLUMN].astype(str).str.extract(r"(\d+)")[0],
		errors="coerce",
	)
	ordered["_dimension_order"] = dimension_numbers.fillna(9999)

	ordered = ordered.sort_values(
		by=["_dimension_order", DIMENSION_CODE_COLUMN, ATTRIBUTE_COLUMN],
		ascending=[True, True, True],
		kind="stable",
	)
	return ordered.drop(columns=["_dimension_order"])


def build_metrics_text(row):
	labels = {
		"Categoria (nominal ou ordinal)": "Escala de mensuracao",
		"Valor min": "Valor minimo",
		"Valor max": "Valor maximo",
		"Média": "Media aritmetica",
		"Desvio padrão": "Desvio padrao",
		"Moda": "Moda",
		"Frequência": "Distribuicao de frequencias",
	}

	parts = []
	for column in METRIC_COLUMNS:
		label = labels[column]
		parts.append(f"{label}: {safe_text(row.get(column))}")

	return "<br/>".join(parts)


def build_table_for_dimension(dimension_df, content_width, header_style, cell_style):
	rows = [
		[
			Paragraph("Referencial teorico (artigos cientificos)", header_style),
			Paragraph("Variavel observacional (codigo e descricao)", header_style),
			Paragraph("Estatisticas descritivas da variavel", header_style),
		]
	]

	for _, row in dimension_df.iterrows():
		rows.append(
			[
				Paragraph(extract_articles_only(row.get(ASPECT_COLUMN)), cell_style),
				Paragraph(safe_text(row.get(ATTRIBUTE_COLUMN)), cell_style),
				Paragraph(build_metrics_text(row), cell_style),
			]
		)

	column_widths = [content_width * 0.36, content_width * 0.26, content_width * 0.38]
	table = Table(rows, colWidths=column_widths, repeatRows=1)
	table.setStyle(
		TableStyle(
			[
				("GRID", (0, 0), (-1, -1), 0.6, colors.black),
				("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
				("VALIGN", (0, 0), (-1, -1), "TOP"),
				("LEFTPADDING", (0, 0), (-1, -1), 4),
				("RIGHTPADDING", (0, 0), (-1, -1), 4),
				("TOPPADDING", (0, 0), (-1, -1), 3),
				("BOTTOMPADDING", (0, 0), (-1, -1), 3),
			]
		)
	)

	return table


def generate_pdf(df, output_path):
	output_path.parent.mkdir(parents=True, exist_ok=True)
	regular_font, bold_font = resolve_pdf_fonts()

	doc = SimpleDocTemplate(
		str(output_path),
		pagesize=A4,
		leftMargin=12 * mm,
		rightMargin=12 * mm,
		topMargin=12 * mm,
		bottomMargin=12 * mm,
	)

	base_styles = getSampleStyleSheet()
	document_header_main_style = ParagraphStyle(
		"DocumentHeaderMain",
		parent=base_styles["Normal"],
		fontName=bold_font,
		fontSize=11,
		leading=13,
		alignment=1,
	)
	document_header_sub_style = ParagraphStyle(
		"DocumentHeaderSub",
		parent=base_styles["Normal"],
		fontName=regular_font,
		fontSize=9,
		leading=11,
		alignment=1,
	)
	dimension_title_style = ParagraphStyle(
		"DimensionTitle",
		parent=base_styles["Normal"],
		fontName=bold_font,
		fontSize=9.5,
		leading=11,
		alignment=1,
	)
	header_style = ParagraphStyle(
		"HeaderCell",
		parent=base_styles["Normal"],
		fontName=bold_font,
		fontSize=7.2,
		leading=8.4,
	)
	cell_style = ParagraphStyle(
		"BodyCell",
		parent=base_styles["Normal"],
		fontName=regular_font,
		fontSize=6.8,
		leading=8,
	)

	story = []
	for index, line in enumerate(DOCUMENT_HEADER_LINES):
		style = document_header_main_style if index == 0 else document_header_sub_style
		story.append(Paragraph(line, style))
		story.append(Spacer(1, 1.5 * mm))

	story.append(Spacer(1, 5 * mm))
	grouped_dimensions = list(df.groupby([DIMENSION_CODE_COLUMN, DIMENSION_TITLE_COLUMN], sort=False))

	for index, ((dimension_code, dimension_title), dimension_df) in enumerate(grouped_dimensions):
		title_value = safe_text(dimension_title)
		if title_value == "Nao se aplica":
			title_value = safe_text(dimension_code)

		story.append(Paragraph(f"Dimensão: {title_value}", dimension_title_style))
		story.append(Spacer(1, 3 * mm))

		table = build_table_for_dimension(dimension_df, doc.width, header_style, cell_style)
		story.append(table)

		if index < len(grouped_dimensions) - 1:
			story.append(Spacer(1, 6 * mm))

	doc.build(story)
	return len(grouped_dimensions)


def main():
	parser = argparse.ArgumentParser(description="Gera PDF formal da tabela de atributos.")
	parser.add_argument(
		"--input",
		help="Caminho do CSV de entrada. Se omitido, usa o CSV padrao em data/processed.",
	)
	parser.add_argument(
		"--output",
		help="Caminho do PDF de saida. Se omitido, usa data/processed/04_feature_statistics_table_formal.pdf.",
	)
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parents[1]
	input_path = resolve_input_path(project_root, args.input)
	output_path = resolve_output_path(project_root, args.output)

	if not input_path.exists():
		raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

	df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
	required_columns = [DIMENSION_CODE_COLUMN, DIMENSION_TITLE_COLUMN, ASPECT_COLUMN, ATTRIBUTE_COLUMN, *METRIC_COLUMNS]
	missing_columns = [column for column in required_columns if column not in df.columns]
	if missing_columns:
		raise ValueError(f"Colunas obrigatorias ausentes no CSV: {missing_columns}")

	ordered_df = order_dataframe_by_dimension(df)
	total_dimensions = generate_pdf(ordered_df, output_path)

	print("PDF gerado com sucesso.")
	print(f"Entrada: {input_path}")
	print(f"Saida: {output_path}")
	print(f"Total de atributos: {len(df)}")
	print(f"Total de dimensoes no PDF: {total_dimensions}")


if __name__ == "__main__":
	main()
