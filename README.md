#  Reconhecimento de Dígitos com Árvore de Decisão

Este repositório contém uma aplicação de Inteligência Artificial para classificação de dígitos manuscritos (0-9). O diferencial deste projeto é a utilização de uma Random Forest "Caseira", construída do zero sobre o algoritmo ID3 (Interactive Dichotomizer 3).

##  Equipe
* **Nome Completo 1** - Allex Lemos de Souza Pinheiro
* **Nome Completo 2** - Mateus da Silva Barreto
* **Nome Completo 3** - Vitor Araújo Andrade

##  Tecnologias e Bibliotecas
* **Linguagem:** Python 3.10+
* **Interface:** Streamlit
* **Processamento de Dados:** Pandas & Numpy
* **Visualização gráfica:** Matplotlib
* **Dataset**: Scikit-Learn (apenas para carga do dataset Digits)

## Estrutura do Repositório
* IDF3.py: Implementação do algoritmo de árvore de decisão recursivo (Entropia e Ganho de Informação).
* Random_Forest_Caseiro.py: Lógica de criação da floresta, amostragem de dados e sorteio de atributos.
* treinamento.py: Script de avaliação de métricas (Precision, Recall, F1-Score) via terminal.
* app.py: Dashboard interativo para diagnóstico e visualização da IA.
* relatorio-arv-decisao: relatorio em pdf do projeto

##  Como Executar

### 1. Clonar o repositório
```bash
git clone [https://github.com/SEU_USUARIO/NOME_DO_REPO.git]
cd NOME_DO_REPO

```

### 2. Configurar Ambiente Virtual

```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt

```

### 4. Rodar a Aplicação Web

```bash
streamlit run app.py

```

### 4.1 Rodar o CLI de Avaliação de Precisão

```bash
python treinamento.py

```

##  O Algoritmo

O modelo evoluiu de uma árvore simples para uma floresta otimizada, atingindo os seguintes marcos:
* ID3 Simples: ~87% de acurácia.

* Random Forest (100 árvores): 97.50% de acurácia.

## Hugging Face do projeto

https://huggingface.co/spaces/lexlemos/letter_recognition

