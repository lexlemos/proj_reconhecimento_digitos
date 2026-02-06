#  Reconhecimento de Dígitos com Árvore de Decisão

Este repositório contém uma aplicação de Inteligência Artificial para classificação de dígitos manuscritos (0-9), utilizando o algoritmo **Decision Tree (CART)**. O foco do projeto é a **IA Explicável (XAI)**, permitindo rastrear cada decisão do modelo.

##  Equipe
* **Nome Completo 1** - Allex Lemos de Souza Pinheiro
* **Nome Completo 2** - XXX
* **Nome Completo 3** - XXX

##  Tecnologias e Bibliotecas
* **Linguagem:** Python 3.10+
* **IA/ML:** Scikit-Learn (DecisionTreeClassifier)
* **Interface:** Streamlit
* **Processamento de Dados:** Pandas & Numpy
* **Visualização:** Matplotlib

##  Como Executar

### 1. Clonar o repositório
```bash
git clone [https://github.com/SEU_USUARIO/NOME_DO_REPO.git](https://github.com/SEU_USUARIO/NOME_DO_REPO.git)
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

### 4. Rodar a Aplicação

```bash
streamlit run app.py

```

##  O Algoritmo

O modelo utiliza o critério de **Entropia** para calcular o Ganho de Informação em cada nó. A interface permite visualizar o "Mapa de Atenção", numerando os pixels que a árvore utilizou para chegar ao veredito final.

```

