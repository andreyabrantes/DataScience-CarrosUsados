# 🚗 Previsão e Análise de Preços de Carros Usados no Brasil

> Trabalho da disciplina de **Data Science** — 6º Período  
> Aplicação de técnicas de Machine Learning para previsão de preços de veículos usados no mercado brasileiro.

---

## 📋 Sumário

- [Descrição do Projeto](#-descrição-do-projeto)
- [Dataset](#-dataset)
- [Análise Exploratória dos Dados (AED)](#-análise-exploratória-dos-dados-aed)
- [Limpeza e Preparação dos Dados](#-limpeza-e-preparação-dos-dados)
- [Seleção de Variáveis](#-seleção-de-variáveis)
- [Modelos de Machine Learning](#-modelos-de-machine-learning)
- [Resultados](#-resultados)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Como Executar](#-como-executar)
- [Estrutura do Projeto](#-estrutura-do-projeto)

---

## 📌 Descrição do Projeto

O mercado secundário de automóveis no Brasil é altamente fragmentado, com preços variando significativamente com base em fatores como:

- Idade do veículo
- Quilometragem rodada
- Cilindrada do motor (cc)
- Tipo de combustível
- Tipo de transmissão
- Marca e modelo

Este projeto constrói um **pipeline completo de Machine Learning e Análise Exploratória de Dados (AED)** com o objetivo de:

1. Realizar uma análise exploratória abrangente dos dados do mercado de carros usados
2. Executar limpeza, extração de features e pré-processamento dos dados
3. Criar features específicas do domínio (ex: Idade do Veículo, Marca)
4. Treinar, comparar e avaliar múltiplos modelos de ML (Linear, baseados em Árvore, Ensembles e KNN)
5. Identificar o modelo com melhor desempenho para previsão de preços

---

## 📂 Dataset

### Arquivo: `Carros_Usados_Brasil.csv`

O dataset contém **30 registros** de veículos usados no mercado brasileiro, com as seguintes colunas:

| Coluna | Tipo | Descrição |
|---|---|---|
| `Car Name` | string | Nome completo do veículo (Marca + Modelo) |
| `Model Year` | int | Ano de fabricação do veículo |
| `Mileage` | int | Quilometragem total percorrida (km) |
| `Fuel Type` | string | Tipo de combustível (Flex, Gasolina, Diesel, Híbrido) |
| `Engine Capacity` | int | Cilindrada do motor em cc |
| `Transmission` | string | Tipo de câmbio (Manual ou Automático) |
| `Price` | int | **Variável alvo** — Preço de venda em R$ |

### Exemplos de registros

| Veículo | Ano | Km | Combustível | Cilindrada | Câmbio | Preço (R$) |
|---|---|---|---|---|---|---|
| Chevrolet Onix Plus | 2021 | 35.000 | Flex | 1000 | Manual | 68.500 |
| Hyundai Creta | 2022 | 22.000 | Flex | 2000 | Automático | 112.000 |
| Toyota Hilux | 2020 | 55.000 | Diesel | 2800 | Automático | 210.000 |
| BMW Série 3 | 2021 | 22.000 | Gasolina | 2000 | Automático | 260.000 |

### Estatísticas Descritivas

| Variável | Mínimo | Média | Máximo |
|---|---|---|---|
| Ano do Modelo | 2018 | 2021 | 2023 |
| Quilometragem | 12.000 | 36.000 | 75.000 |
| Cilindrada (cc) | 1000 | 1640 | 3200 |
| Preço (R$) | 42.000 | 114.617 | 260.000 |

---

## 🔍 Análise Exploratória dos Dados (AED)

A análise exploratória foi realizada para entender a distribuição e as relações entre as variáveis do dataset. As principais etapas incluíram:

- **Visualização das primeiras linhas** com `df.head()` para entender a estrutura dos dados
- **Análise de tipos e estrutura** com `df.info()` para verificar os tipos de dados de cada coluna
- **Estatísticas descritivas** com `df.describe()` para obter medidas de tendência central e dispersão
- **Verificação de valores ausentes** com `df.isnull().sum()` — nenhum valor nulo foi encontrado no dataset
- **Verificação de duplicados** com `df.duplicated().sum()` — nenhum registro duplicado
- **Verificação de inconsistências** — valores únicos das categorias, faixas, negativos e limites de plausibilidade
- **Distribuições** de preço, quilometragem, cilindrada e idade; **relações** de preço com km, cilindrada e idade; **preço por** combustível, câmbio e marca; **correlações** (Pearson e Spearman)

### Principais Observações

- O dataset não apresentou **valores ausentes** nem **duplicados**; não há valores negativos ou impossíveis
- Inconsistência corrigida: o arquivo usa `Gasoline`, mas o mapeamento de tradução só cobria `Petrol`, deixando 2 veículos com a categoria em inglês
- A maioria dos veículos utiliza combustível **Flex** (22 de 30)
- Veículos com motor **Diesel** apresentam preços mais elevados (ex: Toyota Hilux, Ford Ranger)
- Veículos com câmbio **Automático** apresentam preços superiores aos de câmbio Manual (mediana ≈ R$ 118 mil vs ≈ R$ 55 mil)
- **Cilindrada** tem a correlação mais forte com o preço (r ≈ 0,83); quilometragem e idade quase não se correlacionam com o preço no conjunto completo, mas entre os veículos Flex a correlação é negativa (≈ −0,64)

---

## 🛠️ Limpeza e Preparação dos Dados

### Etapas Realizadas

1. **Remoção de artefatos de indexação**: Verificação e remoção da coluna `Unnamed: 0`, caso presente
2. **Renomeação de colunas**: Tradução dos nomes das colunas para o português e adaptação ao contexto brasileiro
3. **Tradução de valores categóricos**:
   - `Fuel Type`: `Gasoline`/`Petrol` → `Gasolina`, `Hybrid` → `Híbrido`, `CNG` → `GNV`, `Electric` → `Elétrico`, `LPG` → `GLP`
   - `Transmission`: `Automatic` → `Automático`, `Manual` → `Manual`
4. **Remoção de registros com variável alvo ausente**: Linhas sem valor de `Preco_BRL` foram descartadas
5. **Extração da Marca**: A coluna `Marca` foi criada a partir do primeiro token de `Nome_Carro`
6. **Criação da feature `Idade_Veiculo`**: Calculada como `2026 - Ano_Modelo`

### Resultado após Limpeza

- **Total de registros**: 30
- **Total de colunas**: 9 (incluindo as novas features criadas)
- **Valores ausentes**: 0

---

## 📊 Seleção de Variáveis

Para o treinamento dos modelos, as variáveis foram divididas em:

### Features Categóricas
| Feature | Descrição |
|---|---|
| `Tipo_Combustivel` | Tipo de combustível do veículo |
| `Transmissao` | Tipo de câmbio (Manual/Automático) |
| `Marca` | Marca do fabricante |

### Features Numéricas
| Feature | Descrição |
|---|---|
| `Cilindrada` | Tamanho do motor em cc |
| `Quilometragem` | Distância total percorrida (km) |
| `Idade_Veiculo` | Idade calculada em anos (2026 - Ano_Modelo) |

### Variável Alvo
- **`Preco_BRL`**: Preço de venda do veículo em Reais (R$)
  - Transformação logarítmica aplicada (`log1p`) para normalizar a distribuição

### Pré-processamento
- **Features numéricas**: Normalização com `StandardScaler`
- **Features categóricas**: Codificação com `OneHotEncoder` (com `handle_unknown='ignore'`)
- **Divisão treino/teste**: 80% treino (24 amostras) / 20% teste (6 amostras), com `random_state=42`

---

## 🤖 Modelos de Machine Learning

Foram treinados e avaliados **7 modelos de regressão**, todos utilizando um pipeline com pré-processamento integrado:

| Modelo | Descrição |
|---|---|
| Regressão Linear | Modelo linear simples |
| Ridge | Regressão linear com regularização L2 (α=1.0) |
| Lasso | Regressão linear com regularização L1 (α=0.001) |
| Árvore de Decisão | Árvore de decisão com profundidade máxima de 10 |
| Random Forest | Ensemble de 100 árvores de decisão |
| Gradient Boosting | Ensemble com boosting, 100 estimadores |
| K-Nearest Neighbors (KNN) | Regressão por vizinhos mais próximos (k=5) |

### Otimização de Hiperparâmetros — KNN

Para o modelo KNN, foi realizada uma busca em grade (`GridSearchCV`) com validação cruzada de 5 folds, testando:

- **`n_neighbors`**: de 1 a 14 vizinhos
- **`weights`**: `uniform` ou `distance`

**Melhores parâmetros encontrados**: `n_neighbors=4`, `weights='distance'`

---

## 📈 Resultados

### Comparação dos Modelos (Conjunto de Teste, 6 veículos)

| Modelo | R² (R$) | RMSE (R$) | MAE (R$) | R² (log) | RMSE (log) | MAE (log) |
|---|---|---|---|---|---|---|
| Random Forest | 0.8134 | 16.821 | 14.708 | 0.7978 | 0.2110 | 0.1804 |
| Gradient Boosting | 0.7773 | 18.377 | 16.458 | 0.7876 | 0.2162 | 0.1887 |
| Ridge | 0.7537 | 19.325 | 16.135 | 0.8116 | 0.2037 | 0.1744 |
| KNN (Otimizado) | 0.6603 | 22.696 | 20.578 | 0.6614 | 0.2730 | 0.2484 |
| KNN (Original, k=5) | 0.5677 | 25.605 | 24.571 | 0.5349 | 0.3200 | 0.3003 |
| Regressão Linear | 0.5662 | 25.648 | 23.204 | 0.6971 | 0.2582 | 0.2473 |
| Lasso | −0.5422 | 48.360 | 33.520 | 0.5328 | 0.3207 | 0.2819 |
| Árvore de Decisão | −1.0271 | 55.444 | 38.417 | 0.3735 | 0.3714 | 0.3200 |

> **Métricas**: R² (quanto maior, melhor), RMSE e MAE (quanto menor, melhor). As colunas "(R$)" usam as previsões convertidas de volta para reais com `np.expm1`; as colunas "(log)" são calculadas sobre `log1p(preço)`, a escala de treino.

### Validação Cruzada Complementar (5-fold, 30 veículos)

| Modelo | R² log médio ± desvio | MAE fora da amostra (R$) |
|---|---|---|
| Ridge | 0.785 ± 0.162 | 19.876 |
| Random Forest | 0.613 ± 0.444 | 24.664 |
| KNN (k=5) | 0.610 ± 0.305 | 25.064 |
| Gradient Boosting | 0.594 ± 0.438 | 26.354 |
| Lasso | 0.648 ± 0.152 | 29.220 |
| Regressão Linear | 0.501 ± 0.264 | 34.149 |
| Árvore de Decisão | 0.312 ± 0.666 | 37.150 |

### Análise dos Resultados

- No conjunto de teste, **Random Forest**, **Gradient Boosting** e **Ridge** tiveram os menores erros (RMSE entre ≈ R$ 16,8 mil e ≈ R$ 19,3 mil, ou seja, erros da ordem de 15–20% do preço)
- Na validação cruzada, o **Ridge** foi o mais estável; Random Forest e Gradient Boosting variaram muito entre folds, logo a vantagem do Random Forest no teste **não se confirma** de forma robusta
- A **otimização do KNN** (k=4, `weights='distance'`, R² de validação cruzada ≈ 0,683) melhorou o modelo em relação ao k=5, mas ele permaneceu abaixo dos três melhores
- **Lasso** e **Árvore de Decisão** tiveram R² negativo em R$, principalmente por um único veículo: o Honda Civic (gasolina, R$ 135 mil), cujo único "par" a gasolina no treino é o BMW Série 3 (R$ 260 mil)
- **Limitação**: com 30 registros e 6 veículos de teste, os resultados têm alta variabilidade e **não devem ser generalizados** para o mercado de carros usados

---

## 🧰 Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.x | Linguagem principal |
| Pandas | — | Manipulação e análise de dados |
| NumPy | — | Operações numéricas |
| Scikit-learn | — | Modelos de ML, pré-processamento e avaliação |
| Matplotlib | — | Visualização de dados |
| Seaborn | — | Visualização estatística |
| Jupyter Notebook | — | Ambiente de desenvolvimento |

---

## ▶️ Como Executar

### Pré-requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Passos

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/andreyabrantes/DataScience-CarrosUsados.git
   cd DataScience-CarrosUsados
   ```

2. **Certifique-se de que o dataset está na mesma pasta** que o notebook:
   ```
   Carros_Usados_Brasil.csv
   used_car_valuation_engine_ptbr.ipynb
   ```

3. **Inicie o Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Abra o arquivo** `used_car_valuation_engine_ptbr.ipynb` e execute as células em ordem.

> O notebook também pode ser executado no **Google Colab** — basta fazer o upload dos arquivos ou montar o Google Drive.

---

## 📁 Estrutura do Projeto

```
DataScience-CarrosUsados-main/
│
├── Carros_Usados_Brasil.csv              # Dataset com 30 registros de carros usados
├── used_car_valuation_engine_ptbr.ipynb  # Notebook principal com toda a análise
└── README.md                             # Este arquivo
```

---

## 📚 Referências
- [Dataset original - Releitura] (https://www.kaggle.com/code/sumedh1507/used-car-valuation-engine)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)


## Equipe
- Andrey Campos
- Nathan Salles
- Cristiano Cordeiro
- Gustavo Ramos
- Julia Scarpi
---

*Projeto desenvolvido para a disciplina de Data Science — UNIFESO, 2026.*
