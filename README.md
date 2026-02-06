# proj_reconhecimento_digitos
Excelente escolha. Manter a organização via terminal é a marca de um desenvolvedor sênior. Vamos estruturar o repositório para que ele seja um "cartão de visitas" técnico.

### 1. Criando o README.md Magnífico

Primeiro, vamos criar o arquivo que descreve o projeto. No seu VS Code, crie o arquivo `README.md` e cole o conteúdo abaixo (ajustando os nomes da sua equipe):

```markdown
#  Reconhecimento de Dígitos com Árvore de Decisão

Este repositório contém uma aplicação de Inteligência Artificial para classificação de dígitos manuscritos (0-9), utilizando o algoritmo **Decision Tree (CART)**. O foco do projeto é a **IA Explicável (XAI)**, permitindo rastrear cada decisão do modelo.

##  Equipe
* **Nome Completo 1** - [Função/RA]
* **Nome Completo 2** - [Função/RA]
* **Nome Completo 3** - [Função/RA]

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

---

### 2. Inicializando o Git via Terminal

Agora, no seu terminal (com o `venv` ativo e dentro da pasta do projeto), siga estes passos:

1.  **Crie o arquivo `.gitignore`** para não subir arquivos desnecessários:
    ```cmd
    echo venv/ > .gitignore
    echo __pycache__/ >> .gitignore
    echo .streamlit/ >> .gitignore
    ```

2.  **Inicialize o repositório local:**
    ```cmd
    git init
    git add .
    git commit -m "Initial commit: IA de reconhecimento de letras com interface explicável"
    ```

---

```
