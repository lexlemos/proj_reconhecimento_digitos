import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

def executar_treinamento():
    # --- [PASSO 1: CARREGAMENTO E PREPARAÇÃO] ---
    # Carregamos o dataset onde cada imagem 8x8 é um vetor de 64 pixels.
    dados = load_digits()
    X = dados.data 
    y = dados.target 

    # Dividimos os dados: 80% para aprender (treino) e 20% para testar (avaliação).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Dataset carregado: {X_train.shape[0]} amostras de treino.")

    # --- [PASSO 2: O ALGORITMO (MAPEADO AO PSEUDOCÓDIGO)] ---
    # O pseudocódigo CART/ID3 inicia aqui. 
    # Definimos o critério de 'entropy' (Entropia) para medir o Ganho de Informação.
    
    # [PSEUDOCÓDIGO]: Criar_No_Decisao(Dados)
    # [PSEUDOCÓDIGO]: Se ganho_informacao < limiar, torna-se Nó Folha.
    modelo = DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=10,        # Limita a altura da árvore (Critério de Parada)
        min_samples_leaf=5,  # Mínimo de amostras por nó (Evita Overfitting)
        random_state=42
    )

    # [PSEUDOCÓDIGO]: Recursão para dividir os nós (Ajuste do modelo)
    modelo.fit(X_train, y_train)

    # [PASSO 3: AVALIAÇÃO DE PERFORMANCE]
    previsoes = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, previsoes)
    
    print("\n--- Resultado do Modelo ---")
    print(f"Acurácia: {acuracia * 100:.2f}%")
    print("\nRelatório de Classificação:\n", classification_report(y_test, previsoes))

    # [PASSO 4: VISUALIZAÇÃO DA ESTRUTURA] 
    plt.figure(figsize=(20,10))
    plot_tree(modelo, filled=True, feature_names=[f"px_{i}" for i in range(64)], class_names=[str(i) for i in range(10)])
    plt.title("Estrutura Hierárquica da Árvore de Decisão")
    plt.savefig("arvore_decisao.png")
    print("\nVisualização da árvore salva como 'arvore_decisao.png'.")

if __name__ == "__main__":
    executar_treinamento()