import math
import random
from collections import Counter

# --- FUNÇÕES DE UTILIDADE ---
# Estas funções ajudam a manipular os dicionários de dados de forma mais limpa,
# evitando repetição de código ao acessar chaves específicas.

def get_classification(example):
    """Retorna o rótulo/classe (ex: o dígito de 0 a 9) daquele exemplo."""
    return example['target']

def get_value(example, attribute):
    """Retorna o valor de um atributo específico (ex: se o px_15 é 0 ou 1)."""
    return example[attribute]

def plurality_value(examples):
    """
    VOTO DE MAIORIA (Empate/Fim de linha): 
    Se a árvore não consegue mais se dividir, ela chuta a classe que mais apareceu
    naquele grupo de exemplos.
    """
    if not examples: return None
    labels = [get_classification(e) for e in examples] 
    return Counter(labels).most_common(1)[0][0]

def all_same_classification(examples):
    """
    TESTE DE PUREZA: Verifica se todos os exemplos no nó atual pertencem à mesma classe.
    Se sim, não há por que continuar dividindo a árvore.
    """
    if not examples: return True
    primeira = get_classification(examples[0])
    return all(get_classification(e) == primeira for e in examples)

def possible_values(attribute, examples):
    """Retorna os caminhos possíveis para um atributo (geralmente [0, 1] em imagens binarizadas)."""
    return list(set(get_value(e, attribute) for e in examples))

# --- MATEMÁTICA ---
# O "cérebro" da árvore, responsável por decidir qual pixel ou forma divide melhor os números.

def entropy(examples):
    """
    ENTROPIA DE SHANNON: Mede o grau de 'bagunça' ou mistura dos dados.
    Quanto mais misturados os números (ex: 50% de '1' e 50% de '8'), maior a entropia.
    """
    if not examples: return 0.0
    labels = [get_classification(e) for e in examples]
    total = len(labels)
    contagem = Counter(labels)
    ent = 0.0
    for c in contagem.values():
        p = c / total
        ent -= p * math.log2(p) # Fórmula: -p * log2(p)
    return ent

def importance(attribute, examples):
    """
    GANHO DE INFORMAÇÃO: Avalia o quão bom é um atributo.
    Ele calcula a entropia atual e subtrai a entropia que teríamos SE dividíssemos
    os dados usando este atributo. O atributo que gerar a maior queda na entropia vence.
    """
    if not examples: return 0.0
    ent_base = entropy(examples)
    total = len(examples)
    
    # Agrupa os exemplos com base no valor do atributo (ex: todos com px_1=0 e px_1=1)
    subconjuntos = {}
    for e in examples:
        v = get_value(e, attribute)
        # setdefault é um truque para criar a lista vazia se a chave não existir
        subconjuntos.setdefault(v, []).append(e)
        
    # Média ponderada da entropia dos novos grupos gerados
    ent_esp = sum((len(s)/total) * entropy(s) for s in subconjuntos.values())
    return ent_base - ent_esp

# --- O ALGORITMO ID3 ---
# Constrói a árvore de decisão de forma recursiva (chamando a si mesma).

def learn_decision_tree(examples, attributes, parent_examples, max_depth=10, depth=0):
    # CASOS BASE (Condições de Parada):
    if not examples:
        return plurality_value(parent_examples) # Evita galho vazio
    
    # Pré-poda: Para se tiver menos de 5 exemplos ou atingiu o limite de profundidade
    if len(examples) < 5 or depth >= max_depth: 
        return plurality_value(examples)
        
    # Para se o grupo já é 100% puro (ex: só tem número '3' aqui)
    elif all_same_classification(examples):
        return get_classification(examples[0])
        
    # Para se não há mais atributos para testar
    elif not attributes:
        return plurality_value(examples)
        
    # CASO RECURSIVO (Ainda pode crescer):
    else:
        # 1. Escolhe o melhor atributo usando a função de Ganho de Informação
        A = max(attributes, key=lambda a: importance(a, examples))
        
        # 2. Cria a 'raiz' deste pedaço da árvore
        tree = {A: {}}
        
        # 3. Para cada valor possível (0 ou 1), cria um ramo
        for v in possible_values(A, examples):
            exs = [e for e in examples if get_value(e, A) == v]
            res_attrs = [attr for attr in attributes if attr != A] # Remove o atributo já usado
            
            # 4. Chama a si mesma para o próximo nível (aumentando a profundidade)
            tree[A][v] = learn_decision_tree(exs, res_attrs, examples, max_depth, depth + 1)
            
        return tree

# --- LÓGICA DA FLORESTA ---
# Transforma uma única árvore ID3 (que é fraca) em uma Random Forest (que é forte).

def treinar_random_forest(dados_treino, atributos, n_arvores=15, profundidade_maxima=10):
    floresta = []
    print(f"Construindo Floresta com {n_arvores} árvores (Profundidade: {profundidade_maxima})...")
    
    for i in range(n_arvores):
        # TÉCNICA 1: BAGGING (Bootstrap Aggregating)
        # Sorteia aleatoriamente 70% dos dados para treinar esta árvore específica.
        # Isso faz com que cada árvore veja dados ligeiramente diferentes.
        amostra = random.choices(dados_treino, k=int(len(dados_treino) * 0.7))
        
        # TÉCNICA 2: FEATURE SELECTION (Subspace Method)
        # Em vez de olhar para os 64 pixels + 4 formas, sorteia no máximo 25 atributos.
        # Isso impede que o pixel mais "forte" domine todas as árvores, forçando a IA a achar outros padrões.
        atrs_sorteados = random.sample(atributos, k=min(25, len(atributos)))
        
        # Treina uma árvore do zero com esses dados e atributos restritos
        arvore = learn_decision_tree(amostra, atrs_sorteados, [], max_depth=profundidade_maxima) 
        
        floresta.append(arvore)
        print(f"  > Árvore {i+1} pronta.")
        
    return floresta
