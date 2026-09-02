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

### Principais Observações

- O dataset não apresentou **valores ausentes** em nenhuma coluna
- A maioria dos veículos utiliza combustível **Flex**, refletindo a realidade do mercado brasileiro
- Veículos com motor **Diesel** tendem a ter preços mais elevados (ex: Toyota Hilux, Ford Ranger)
- Veículos com câmbio **Automático** geralmente apresentam preços superiores aos de câmbio Manual

---

## 🛠️ Limpeza e Preparação dos Dados

### Etapas Realizadas

1. **Remoção de artefatos de indexação**: Verificação e remoção da coluna `Unnamed: 0`, caso presente
2. **Renomeação de colunas**: Tradução dos nomes das colunas para o português e adaptação ao contexto brasileiro
3. **Tradução de valores categóricos**:
   - `Fuel Type`: `Petrol` → `Gasolina`, `Hybrid` → `Híbrido`, `CNG` → `GNV`, `Electric` → `Elétrico`, `LPG` → `GLP`
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

### Comparação dos Modelos (Conjunto de Teste)

| Modelo | R² | RMSE | MAE |
|---|---|---|---|
| **Ridge** | **0.8116** | **0.2037** | **0.1744** |
| **Random Forest** | **0.8034** | **0.2081** | **0.1764** |
| Gradient Boosting | 0.7876 | 0.2162 | 0.1887 |
| Regressão Linear | 0.6971 | 0.2582 | 0.2473 |
| KNN (Otimizado) | 0.6614 | 0.2730 | 0.2484 |
| Lasso | 0.5328 | 0.3207 | 0.2819 |
| KNN (Original, k=5) | 0.5349 | 0.3200 | 0.3003 |
| Árvore de Decisão | 0.3735 | 0.3714 | 0.3200 |

> **Métricas**: R² (quanto maior, melhor), RMSE e MAE (quanto menor, melhor). Valores calculados sobre o log do preço.

### Análise dos Resultados

- **Ridge** e **Random Forest** foram os modelos com melhor desempenho, ambos com R² acima de 0.80
- A **otimização de hiperparâmetros do KNN** melhorou significativamente o desempenho do modelo, elevando o R² de 0.5349 para 0.6614
- A **Árvore de Decisão** apresentou o pior desempenho, possivelmente devido ao overfitting com profundidade máxima de 10 em um dataset pequeno
- O modelo **Lasso** teve desempenho inferior ao Ridge, indicando que a regularização L1 pode ter eliminado features relevantes

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
