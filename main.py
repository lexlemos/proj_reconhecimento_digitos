import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def treinar_floresta():
    digits = load_digits()
    X, y = digits.data, digits.target

    # Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Criando a Random Forest
    # n_estimators=100 significa que teremos 100 árvores votando
    modelo = RandomForestClassifier(
        n_estimators=100, 
        criterion='entropy', 
        random_state=42,
        n_jobs=-1 # Usa todos os núcleos do PC para treinar rápido se tiver pc fraco cuidado com isso
    )

    modelo.fit(X_train, y_train)

    # Avaliação
    previsoes = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, previsoes)
    
    print(f"RESULTADO RANDOM FOREST")
    print(f"Acurácia: {acuracia:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, previsoes))

if __name__ == "__main__":
    treinar_floresta()