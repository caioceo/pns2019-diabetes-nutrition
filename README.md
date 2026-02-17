# Análise dos Padrões Alimentares em Adultos com Diabetes — PNS 2019

## 1. Introdução
Este projeto tem como finalidade analisar a relação entre o diagnóstico de diabetes em adultos e seus padrões alimentares, utilizando dados da Pesquisa Nacional de Saúde (PNS) de 2019, conduzida pelo Instituto Brasileiro de Geografia e Estatística (IBGE). A investigação busca identificar associações entre qualidade e frequência de consumo alimentar e a ocorrência de diabetes na população adulta brasileira.

---

## 2. Público-alvo
O público-alvo do estudo é composto por indivíduos adultos, com idade entre 20 e 59 anos, conforme a faixa etária adotada pelo IBGE para a população adulta. São considerados dois grupos principais:
- Adultos com diagnóstico médico de diabetes.
- Adultos sem diagnóstico médico de diabetes.

---

## 3. Objetivo geral
Avaliar a associação entre padrões alimentares e a presença de diabetes em adultos brasileiros.

---

## 4. Objetivos específicos
- Selecionar variáveis relacionadas ao consumo alimentar, estilo de vida e características antropométricas.
- Construir um subconjunto da base original da PNS contendo apenas atributos relevantes ao tema do estudo.
- Documentar as variáveis selecionadas por meio de um dicionário de dados.
- Comparar os padrões alimentares entre indivíduos diabéticos e não diabéticos.
- Preparar os dados para posterior aplicação de técnicas de Mineração de Dados e Aprendizado de Máquina.

---

## 5. Base de dados
Os dados utilizados são provenientes da Pesquisa Nacional de Saúde (PNS) 2019, disponibilizada pelo Instituto Brasileiro de Geografia e Estatística (IBGE). A base original possui um grande número de variáveis, sendo necessária a realização de um processo de seleção de atributos voltado à temática de nutrição, estilo de vida e diabetes.

---

## 6. Metodologia (etapa atual)
Na etapa atual do projeto, foram realizadas as seguintes atividades:
- Leitura da base original da PNS 2019.
- Construção de um dataset reduzido contendo apenas os atributos selecionados.
- Geração automática de um dicionário de dados em formato Markdown, descrevendo cada variável utilizada.

---

## 7. Considerações metodológicas
Variáveis que representam consequências diretas do diagnóstico de diabetes (por exemplo, orientações médicas recebidas após o diagnóstico) não são utilizadas na construção de modelos preditivos, de modo a evitar o vazamento de informação (target leakage). São priorizadas variáveis que representam exposição alimentar, comportamento e condições físicas dos indivíduos.

---

## 8. Próximas etapas
- Filtragem da população adulta (20 a 59 anos).
- Separação dos grupos de indivíduos diabéticos e não diabéticos.
- Tratamento de valores ausentes.
- Construção de variáveis derivadas, como o Índice de Massa Corporal (IMC) e o tempo semanal de atividade física.
- Aplicação de técnicas de Mineração de Dados e Aprendizado de Máquina para identificação de padrões.

---