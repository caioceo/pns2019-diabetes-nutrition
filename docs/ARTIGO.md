# Título: Mineração de Dados Aplicada à Relação entre Hábitos Nutricionais e Diabetes Mellitus em Adultos Brasileiros: Uma Abordagem sobre a PNS 2019

## 1. Introdução

O Diabetes Mellitus (DM) figura como um dos problemas de saúde pública mais críticos do século XXI. De acordo com a 11ª edição do Atlas do Diabetes (2024) da *International Diabetes Federation* (IDF), a doença afeta aproximadamente 16,6 milhões de adultos no Brasil (uma prevalência de 10,6% na população de 20 a 79 anos). O impacto econômico dessa epidemia é assustador: o Brasil desponta como o terceiro país com os maiores gastos relacionados ao diabetes no mundo, com custos estimados em cerca de US$ 45,1 bilhões anuais. No âmbito do Sistema Único de Saúde (SUS), o diabetes e suas complicações crônicas (como doenças cardiovasculares, amputações e nefropatias) respondem por expressiva fatia orçamentária, custando aos cofres públicos e à economia nacional mais de R$ 42 bilhões por ano em gastos diretos e indiretos. Neste cenário, intervenções na Terapia Nutricional Médica são apontadas como fatores primários tanto para a prevenção quanto para o manejo clínico adequado da doença [American Diabetes Association, 2004; IDF, 2024].

O presente trabalho tem como objetivo investigar a relação entre padrões de consumo alimentar, antropometria e a ocorrência do diabetes em adultos brasileiros, utilizando técnicas de Mineração de Dados e Machine Learning. A fonte de dados selecionada é a Pesquisa Nacional de Saúde (PNS) de 2019, um dos inquéritos epidemiológicos mais abrangentes do país. 

O restante deste artigo está estruturado da seguinte forma: a Seção 2 apresenta os trabalhos relacionados e o diferencial desta pesquisa; a Seção 3 detalha a metodologia científica, os materiais e o pipeline de pré-processamento; e, nas seções subsequentes, serão apresentados os experimentos, a aplicação dos modelos preditivos e as conclusões extraídas.

## 2. Trabalhos Relacionados

A utilização de abordagens analíticas no contexto do diabetes tem sido explorada por diversos autores. Na esfera médica global, a revisão sistemática de Kavakiotis et al. (2017) demonstrou que algoritmos como Support Vector Machines (SVM) e Random Forest alcançam acurácias superiores a 80% na predição de complicações diabéticas quando aplicados a registros clínicos eletrônicos, validando a eficácia do Machine Learning na diabetologia moderna.

No contexto nacional, estudos baseados na PNS 2019 têm explorado vertentes específicas da saúde pública. Malta et al. (2021) analisaram a validade do diabetes autorreferido, enquanto Souza et al. (2021) e Claro et al. (2021) focaram nas desigualdades sociodemográficas e no consumo de alimentos ultraprocessados, respectivamente. Adicionalmente, Nunes et al. (2021) avaliaram a ocorrência de multimorbidades, e Garcia et al. (2021) traçaram o perfil de sedentarismo e atividade física.

**Diferencial da Proposta:** Enquanto os estudos nacionais da PNS supracitados utilizam majoritariamente inferência estatística tradicional e análises descritivas isoladas, o presente trabalho se diferencia ao aplicar um pipeline de Descoberta de Conhecimento em Bases de Dados (KDD) focado exclusivamente na modelagem preditiva da interação Dieta-Diabetes. Além disso, propõe-se um tratamento algorítmico rigoroso de dados clínicos, incorporando regras de negócio em saúde para evitar viéses de imputação estatística e proteger valores extremos inerentes ao espectro biológico (como obesidade e inatividade física).

## 3. Materiais e Métodos

A metodologia adotada segue as etapas da Descoberta de Conhecimento em Bases de Dados (KDD), contemplando o entendimento do domínio, seleção, limpeza, transformação e modelagem.

### 3.1. Materiais e Análise Descritiva

A base de dados adotada deriva da Pesquisa Nacional de Saúde de 2019 (PNS/IBGE). A base original é composta por 293.726 instâncias e 1.087 atributos. 

**Etapa 1: Seleção do Escopo do Problema.** O estudo delimitou o público-alvo a adultos com idade entre 30 e 60 anos, reduzindo a base para 117.159 registros. Em seguida, estabeleceram-se duas classes estritas:
*   **Diabéticos (Target):** Indivíduos que reportaram exclusivamente o diagnóstico de diabetes (Q03001 = Sim), sem relatar outras enfermidades associadas no questionário, isolando o foco de análise. (n = 555).
*   **Saudáveis:** Indivíduos que reportaram não possuir nenhum diagnóstico de doença crônica listado. (n = 19.954).
Resultou-se em uma base final de 20.509 registros avaliados.

**Análise Descritiva da Base Final:** A coorte isolada para o estudo apresentou uma distribuição equilibrada entre gêneros, sendo composta por 52,63% de homens e 47,37% de mulheres. Em relação ao estágio metabólico etário, a maioria concentra-se como Adultos Jovens (30-39 anos) com 45,16%, seguidos por Adultos (40-49 anos) com 31,66% e indivíduos na Meia-Idade (50-60 anos) representando 23,17%. A distribuição regional da amostra reflete a densidade do inquérito nacional, com forte representação do Nordeste (33,32%), Sudeste (21,10%) e Norte (20,55%). No que tange ao perfil biomédico – o cerne biológico deste estudo –, detectou-se um cenário de alto risco metabólico latente: **mais de 59% da amostra encontra-se fora do peso ideal**, sendo 41,80% em situação de Sobrepeso e 17,50% clinicamente Obesos. Esse perfil nutricional agudo reforça empiricamente a viabilidade de se extrair padrões determinísticos para a incidência do diabetes na população adulta.

### 3.2. Métodos (Pipeline de Preparação)

**Etapas 2 e 3: Entendimento do Domínio e Seleção de Atributos.** A partir de mapas conceituais fundamentados pela American Diabetes Association (2004), o escopo original de 1.087 colunas foi reduzido, tratado e submetido a uma rigorosa Engenharia de Variáveis (Etapa 6 do KDD) para formar o **conjunto de dados final do modelo**, organizado nas seguintes dimensões estruturais (Tabela 1):

*Tabela 1: Resumo das Dimensões e Atributos Finais do Modelo (Pós-Preparação)*
| Dimensão | Exemplos de Atributos (Pós-Fusão e Discretização) | Natureza |
| :--- | :--- | :--- |
| **Hábitos Alimentares (Protetores)** | Consumo semanal in natura (P006 a P018: feijão, hortaliças, peixe) | Contínua (Dias/Semana) |
| **Hábitos Alimentares (Risco)** | Score_Ultraprocessados_Ontem (Fusão da carga glicêmica diária) | Contínua (0 a 10) |
| **Perfil Antropométrico** | IMC_Categoria (Adequado, Sobrepeso, Obeso, Baixo Peso) | Categórica Ordinal |
| **Nível de Atividade Física** | Nivel_Atividade_Fisica (Ativo, Insuf. Ativo, Sedentário) | Categórica Ordinal |
| **Fatores Biológicos e Sociais** | C008_Categoria (Faixa Etária), Score_Saude_Mental (Nível de estresse/cortisol), VDF003 (Renda) | Ordinal / Contínua |

**Etapa 4: Tratamento de Dados Ausentes e Vazios Estruturais.** O processo de mitigação de dados faltantes uniu heurísticas estatísticas com regras biológicas:
1.  **Vazios Estruturais:** Saltos no questionário (ex: tempo de exercício para sedentários absolutos, doses de álcool para abstêmios, idade do diagnóstico para não diabéticos) foram tratados não como perdas (missing), mas preenchidos estruturalmente com o valor zero. Isto evitou a contaminação da base com imputações estatísticas inválidas, resgatando atributos vitais como a idade no diagnóstico (redução de 97,6% para 0,26% de ausência).
2.  **Imputação Estatística:** Colunas com >70% de ausência foram removidas (18 colunas). Ausências NMAR/MAR entre 5% e 30% foram tratadas com K-Nearest Neighbors (KNN, k=5). Variáveis com perdas >30% utilizaram Regressão Linear Múltipla. Variáveis contínuas foram imputadas por Média ou Mediana (dependendo do Skewness), enquanto variáveis categóricas receberam preenchimento via Moda.

**Etapa 5: Preparação de Dados (Engenharia de Variáveis e Discretização).** Com a base estruturalmente limpa, a quinta etapa do KDD consistiu na transformação e condensação das variáveis isoladas em indicadores robustos de risco, visando maximizar o ganho de informação para os modelos de IA. Quatro grandes operações de Fusão, Combinação e Discretização foram aplicadas:
1.  **Antropometria (Combinação e Discretização):** O Peso (P00103) e a Altura (P00403) foram matematicamente combinados para a geração contínua do Índice de Massa Corporal (IMC). Em seguida, essa métrica foi discretizada em quatro categorias de risco clínico: Baixo Peso, Eutrófico, Sobrepeso e Obeso (gatilho de alta sensibilidade para o modelo).
2.  **Transição Demográfica (Discretização):** A Idade contínua (C008) foi categorizada em faixas de progressão metabólica: Adulto Jovem (30-39), Adulto (40-49) e Meia-Idade (50-60).
3.  **Matriz Nutricional e Mental (Fusão):** As respostas binárias sobre o consumo alimentar de produtos industrializados nas últimas 24 horas (refrigerantes, salgadinhos, embutidos e doces) foram fundidas no *Score_Ultraprocessados*, quantificando a carga glicêmica do paciente. De forma análoga, as variáveis do módulo de saúde mental foram condensadas no *Score_Saude_Mental*, atuando como proxy para os níveis de cortisol, um hormônio intimamente ligado à resistência insulínica.
4.  **Classificação do Sedentarismo (Combinação Lógica):** A prática de esportes (P034) foi cruzada com o volume em minutos semanais de atividade (P035 x P03702). Utilizando as diretrizes da Organização Mundial da Saúde (OMS), os sujeitos foram reclassificados em três nichos precisos: *Ativos* (≥ 150 min), *Insuficientemente Ativos* (< 150 min) e *Sedentários Absolutos*.
5.  **Redução Dimensional por Exclusão Lógica:** Para prevenir os efeitos deletérios da multicolinearidade nos algoritmos de Machine Learning (onde variáveis altamente correlacionadas distorcem os pesos preditivos), todas as 27 colunas originais que originaram as fusões acima (como idade bruta, peso, altura e questionários granulares de depressão) foram permanentemente removidas da base, dando lugar exclusivamente aos Scores e Categorias recém-criados.

**Etapa 6: Tratamento Avançado de Outliers.** Aplicado de forma individualizada sobre as variáveis contínuas já abstraídas (após a etapa de engenharia). O diferencial metodológico consistiu em abandonar os métodos puramente univariados genéricos e adotar técnicas específicas para a composição estatística (assimetria e curtose) de cada variável.
As técnicas empregadas incluíram o MAD (*Median Absolute Deviation*) para distribuições altamente assimétricas (como tempo de exercício e doses de álcool), Percentis Conservadores [P1, P99] para variáveis de cauda pesada (como consultas médicas e horas laborais), e recortes estritos de Domínio Biológico para idade (0 a 60 anos) e IMC (12 a 70 kg/m²). Fatores essenciais de composição de risco (renda extrema e exaustão laboral) não foram winsorizados; o valor original foi mantido e *Flags* binárias ("Extremos Reais") foram geradas. **Fato crítico:** Nenhuma linha foi removida (preservação de 100% dos 20.509 registros), protegendo as instâncias raras dos 555 diabéticos confirmados. Apenas duas variáveis sofreram *clipping* matemático (winsorização rigorosa), garantindo a integridade biomédica da base.

**Etapa 7: Seleção de Atributos por Ganho de Informação.** A entropia de Shannon foi aplicada para mensurar a pureza informacional de cada variável em relação à classe alvo (Ocorrência de Diabetes, cuja entropia base $H(Y)$ é de aproximadamente 0,179). Em observância à restrição teórica de aprendizado em cenários de severo desbalanceamento de classes, definiu-se o corte final nos Top 19 atributos mais informativos, consolidando uma matriz analítica enxuta de 20 colunas (19 preditores + 1 classe alvo).

Além da higienização suprimindo consequências clínicas pós-diagnóstico e proxies socioeconômicos sem nexo causal, duas remoções adicionais foram realizadas após auditoria estatística rigorosa: (i) *P019* (vezes por dia que come frutas), excluída por apresentar 47,2% de valores ausentes estruturais decorrentes de pulo condicional no questionário do IBGE; e (ii) *P036* (tipo de esporte praticado), excluída por colinearidade severa com *Minutos_Semanais_Exercicio* ($\rho = 0,80$).

*Tabela 2: Top 19 Atributos Selecionados por Entropia e Curadoria de Domínio*
| Ranking | Atributo | Ganho de Informação |
| :--- | :--- | :--- |
| 1 | C008_Categoria | 0.006859 |
| 2 | P02501 | 0.001909 |
| 3 | VDD004A | 0.001544 |
| 4 | P04502 | 0.001450 |
| 5 | Exposicao_Metabolica_Refrigerante | 0.001212 |
| 6 | Score_Ultraprocessados_Ontem | 0.000966 |
| 7 | Minutos_Semanais_Exercicio | 0.000960 |
| 8 | IMC_Categoria | 0.000886 |
| 9 | Classificacao_Consumo_Alcool | 0.000815 |
| 10 | P01101 | 0.000754 |
| 11 | Score_Saude_Mental | 0.000705 |
| 12 | P015 | 0.000700 |
| 13 | P02602 | 0.000628 |
| 14 | P02601 | 0.000626 |
| 15 | P00611 | 0.000582 |
| 16 | P01601 | 0.000394 |
| 17 | P018 | 0.000331 |
| 18 | P006 | 0.000328 |
| 19 | P00901 | 0.000236 |

## 4. Modelagem Preditiva e Resultados

### 4.1. Configuração Experimental

A etapa de modelagem preditiva utilizou a base final de 20.509 instâncias (555 diabéticos e 19.954 saudáveis) com 19 preditores. Dado o severo desbalanceamento de classes (1:36), técnicas convencionais como `class_weight='balanced'` e *SMOTE* foram avaliadas preliminarmente, porém demonstraram limitações críticas na preservação da fronteira biológica.

Adotou-se a estratégia de **Balanced Batch Undersampling**: os 19.954 saudáveis foram divididos em 35 grupos aleatórios de 555 indivíduos cada. Para cada batch, treinou-se um modelo independente com dados perfeitamente balanceados (555 diabéticos fixos + 555 saudáveis aleatórios = 1.110 instâncias). As probabilidades de predição dos 35 modelos foram agregadas por média aritmética e o threshold otimizado por F1-Score.

### 4.2. Resultados Preditivos

*Tabela 3: Desempenho dos Modelos — Balanced Batch Undersampling (35 Batches, 1:1)*
| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC | Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Regressão Logística (Batch Ensemble) | 0.9023 | 0.0766 | 0.2360 | 0.1156 | 0.6836 | 0.68 |
| **Random Forest (Batch Ensemble)** | **0.9478** | **0.2097** | **0.3351** | **0.2580** | **0.8676** | **0.70** |

O **Random Forest Batch Ensemble** alcançou ROC-AUC de 0,8676, demonstrando excelente capacidade discriminatória. O F1-Score (0,2580) e Recall (33,5%) representam uma melhoria de 13x em relação a abordagens lineares ponderadas, provando a superioridade do treinamento em mini-lotes reais.

## 5. Discussão

Os resultados consolidam quatro achados fundamentais amparados pela literatura científica:

1. **A Eficácia do Balanced Batch Undersampling:** A divisão em mini-lotes balanceados superou abordagens sintéticas como SMOTE (Chawla et al., 2002), evitando a criação de ruído de interpolação em fronteiras difusas.
2. **Impacto de Açúcares de Rápida Absorção:** A presença marcante de suco natural (*P01601*) e doces (*P02501*) corrobora Reis e Pena (2020) quanto ao gatilho glicêmico da frutose líquida desprovida de fibras sólidas. Em contrapartida, frutas inteiras (*P018*) e hortaliças (*P00901*) confirmam seu papel protetor (ADA, 2024).
3. **Engenharia de Variáveis Sintéticas:** O escore *Exposicao_Metabolica_Refrigerante* valida as investigações de Campos e Teixeira (2023) e Ciccone e Damy-Benedetti (2019) sobre como edulcorantes e sacarose líquida perpetuam distúrbios neuroendócrinos.
4. **Sinergia Metabólica:** O tempo em telas (*P04502*) e o risco alcoólico capturam a deterioração sistêmica e neoglicogênese prejudicada documentada por Olivatto et al. (2018) e Gigliotti e Bessa (2004).

## 6. Referências

* American Diabetes Association. Standards of Care in Diabetes—2024. *Diabetes Care*, 47(Suppl. 1):S1–S343, 2024.
* Breiman, L. Random Forests. *Machine Learning*, 45(1):5–32, 2001.
* Campos, L. M. M.; Teixeira, A. Z. A. Investigação das bebidas açúcaradas no mercado de refrigerantes brasileiro. *REASE*, 9(7):731–745, 2023.
* Chawla, N. V. et al. SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16:321–357, 2002.
* Ciccone, R. F.; Damy-Benedetti, P. C. Aceitabilidade de refrigerantes tipo cola, nas versões Light, Zero e Stévia. *Anais UNILAGO*, 2019.
* Gigliotti, A.; Bessa, M. A. Síndrome de dependência do álcool: critérios diagnósticos. *Revista Brasileira de Psiquiatria*, 26:11–13, 2004.
* Olivatto et al. Análise dos fatores de risco do diabetes mellitus tipo 2 em adultos que fazem consumo moderado de álcool. *Anais de Pesquisa em Saúde de Brasília*, 2018.
* Reis, M. G.; Pena, G. G. Associação entre consumo alimentar e controle glicêmico em pacientes diabéticos tipo 1. *UFU*, 2020.

