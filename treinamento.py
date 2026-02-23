import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from Random_Forest_Caseiro import treinar_random_forest, learn_decision_tree

def extrair_todas_as_formas(pixels_64):
    """
    ENGENHARIA DE ATRIBUTOS: 
    Transforma o vetor de 64 pixels em uma matriz 8x8 e procura padrões geométricos.
    Isso ajuda a árvore de decisão a entender conceitos como "curva" ou "reta".
    """
    matriz = pixels_64.reshape(8, 8)
    caracs = {}

    # 1. Reta no Topo: Verifica se existem 3 pixels seguidos na linha superior
    # Útil para identificar o topo do número '7' ou '5'.
    reta = 0
    for c in range(6):
        if matriz[0, c] == 1 and matriz[0, c+1] == 1 and matriz[0, c+2] == 1:
            reta = 1; break
    caracs['f_reta_topo'] = reta

    # 2. Curva na Base: Soma pixels na parte inferior central
    # Útil para identificar a base arredondada do '0', '6' ou '8'.
    curva = 0
    if np.sum(matriz[6:, 2:6]) > 2: curva = 1
    caracs['f_curva_base'] = curva

    # 3. Cruzamentos: Procura pixels que têm vizinhos em várias direções (como um +)
    # Útil para identificar o centro de um '8' ou o meio de um '4'.
    cruz = 0
    for r in range(1, 7):
        for c in range(1, 7):
            if matriz[r,c] == 1 and (matriz[r-1,c]+matriz[r+1,c]+matriz[r,c-1]+matriz[r,c+1]) >= 3:
                cruz += 1
    caracs['f_muitos_cruzamentos'] = 1 if cruz > 2 else 0

    # 4. Centro Preenchido: Verifica o miolo da imagem (matriz 2x2 central)
    # Números como '0' e '8' tendem a ter o centro vazio, enquanto '1' ou '4' podem ter preenchimento.
    caracs['f_centro_preenchido'] = 1 if np.sum(matriz[3:5, 3:5]) >= 2 else 0
    
    return caracs

def prever_individual(no, ex):
    """
    INFERÊNCIA (Navegação na Árvore):
    Função recursiva que percorre o dicionário da árvore até chegar em um valor (folha).
    """
    # Se o 'no' não for um dicionário, chegamos na folha (o número previsto)
    if not isinstance(no, dict): return no
    
    # Pega o atributo que esta parte da árvore quer testar (ex: 'px_34' ou 'f_reta_topo')
    atrib = list(no.keys())[0]
    valor = ex.get(atrib)
    
    # Se o valor do teste não existir no caminho da árvore, tenta um fallback
    if valor not in no[atrib]:
        opcoes = [v for v in no[atrib].values() if not isinstance(v, dict)]
        return opcoes[0] if opcoes else None
        
    # Continua descendo na árvore
    return prever_individual(no[atrib][valor], ex)

def executar():
    """
    PIPELINE PRINCIPAL:
    Carrega dados -> Processa -> Treina -> Avalia.
    """
    # 1. Carregamento e Binarização (Preto e Branco)
    digits = load_digits()
    X_bin = (digits.data > 8).astype(int)
    y = digits.target
    
    # 2. Preparação do Dataset: Unindo pixels brutos + formas extraídas
    examples = []
    for i in range(len(X_bin)):
        ex = {f"px_{j}": X_bin[i][j] for j in range(64)}
        ex.update(extrair_todas_as_formas(X_bin[i]))
        ex['target'] = y[i]
        examples.append(ex)

    # Lista de todos os "campos" que a IA pode analisar
    atrs = [f"px_{i}" for i in range(64)] + ['f_reta_topo', 'f_curva_base', 'f_muitos_cruzamentos', 'f_centro_preenchido']
    
    # 3. Divisão Treino/Teste
    train, test = train_test_split(examples, test_size=0.2, random_state=42)

    # 4. Treinamento da Floresta (Criando 100 árvores diferentes)
    floresta = treinar_random_forest(train, atrs, n_arvores=100, profundidade_maxima = 12)

    # 5. Avaliação: Cada árvore vota, e a maioria vence
    y_true, y_pred = [], []
    for ex in test:
        votos = [prever_individual(arv, ex) for arv in floresta]
        votos = [v for v in votos if v is not None]
        y_true.append(ex['target'])
        
        # O Counter pega o elemento mais frequente na lista de votos
        if votos:
            vencedor = Counter(votos).most_common(1)[0][0]
            y_pred.append(vencedor)
        else:
            y_pred.append(0)

    # 6. Relatórios de Performance
    print(f"\nAcurácia da Floresta: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(classification_report(y_true, y_pred))

# Garante que o treino só comece se o arquivo for executado diretamente
if __name__ == "__main__":
    executar()
