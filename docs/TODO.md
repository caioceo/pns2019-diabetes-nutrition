# FEITO

1. Para a Introdução (Enriquecer o Critério 1)
Dados Epidemiológicos Oficiais: Pesquise os números do Atlas do Diabetes da IDF (International Diabetes Federation) referentes ao Brasil, ou dados do Ministério da Saúde sobre o custo do diabetes para o SUS.
Por que pesquisar: Colocar um parágrafo dizendo "O diabetes afeta X milhões de brasileiros e consome Y bilhões do orçamento de saúde" dá um peso gigantesco à justificativa do seu projeto para o professor.

# FUTURO

2. Para o Balanceamento de Classes (Etapa 8 - Critério 20)
O Problema: Nossa base tem 19.954 saudáveis e apenas 555 diabéticos (cerca de 2.7%). Se jogarmos isso no modelo, ele vai simplesmente chutar que todo mundo é saudável e terá 97% de "acurácia" (mas errará todos os diabéticos, o que na medicina é trágico).
O que pesquisar: Técnicas de balanceamento como SMOTE (Synthetic Minority Over-sampling Technique) ou SMOTENC (para dados categóricos). Pesquise como o SMOTE cria pacientes diabéticos "sintéticos" para equilibrar a balança antes de treinar o modelo.

# FUTURO

3. Para Seleção de Atributos (Etapa 7 - Critério 19)
O Problema: Temos 94 variáveis. Muitas delas (mesmo sem ausentes) podem ser inúteis para prever diabetes, causando ruído e lentidão.
O que pesquisar: O critério do professor exige Entropia ou Wrappers. Pesquise sobre Information Gain (Ganho de Informação) e RFE (Recursive Feature Elimination). Além disso, estude como algoritmos baseados em árvores (como o Random Forest) conseguem ranquear automaticamente a importância das características (ex: provar matematicamente se comer feijão impacta mais ou menos que beber refrigerante).

# FEITO

4. Engenharia de Variáveis / Discretização (Etapa 6 - Critério 17)
O Problema: O professor exige discretização ou fusão de dados.
O que pesquisar: Como transformar variáveis puramente numéricas em categorias com sentido biológico. Por exemplo:
Pegar Peso e Altura e fundir em uma nova coluna chamada IMC (Índice de Massa Corporal).
Pegar a Idade e discretizar em faixas ("30 a 39 anos", "40 a 49 anos", "50 a 60 anos").
Isso ajuda absurdamente os modelos baseados em regras (Árvores de Decisão) a entenderem o problema.

# FUTURO
5. Interpretabilidade de Modelos Médicos (Critério 26)
O Problema: O critério 26 pede para "explorar e analisar o conhecimento extraído". Na área da saúde, não basta o modelo dizer "vai ter diabetes", o médico quer saber o porquê.
O que pesquisar: SHAP values (SHapley Additive exPlanations) ou Extração de Regras de Árvores de Decisão. O SHAP é uma biblioteca fantástica que diz exatamente o que causou a previsão (ex: "Este paciente foi classificado como diabético porque sua idade aumenta 30% a chance, seu IMC elevado aumenta 40%, mas seu consumo de frutas reduziu a chance em 10%"). Usar um gráfico SHAP no artigo garante a glória do seu trabalho!