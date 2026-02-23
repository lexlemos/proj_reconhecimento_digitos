import math
import random
from collections import Counter

# --- FUNÇÕES DE UTILIDADE ---
def get_classification(example):
    return example['target']

def get_value(example, attribute):
    return example[attribute]

def plurality_value(examples):
    if not examples: return None
    labels = [get_classification(e) for e in examples] 
    return Counter(labels).most_common(1)[0][0]

def all_same_classification(examples):
    if not examples: return True
    primeira = get_classification(examples[0])
    return all(get_classification(e) == primeira for e in examples)

def possible_values(attribute, examples):
    return list(set(get_value(e, attribute) for e in examples))

# --- MATEMÁTICA ---
def entropy(examples):
    if not examples: return 0.0
    labels = [get_classification(e) for e in examples]
    total = len(labels)
    contagem = Counter(labels)
    ent = 0.0
    for c in contagem.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def importance(attribute, examples):
    if not examples: return 0.0
    ent_base = entropy(examples)
    total = len(examples)
    subconjuntos = {}
    for e in examples:
        v = get_value(e, attribute)
        subconjuntos.setdefault(v, []).append(e)
    ent_esp = sum((len(s)/total) * entropy(s) for s in subconjuntos.values())
    return ent_base - ent_esp

# --- O ALGORITMO ID3 ---
def learn_decision_tree(examples, attributes, parent_examples, max_depth=10, depth=0):
    if not examples:
        return plurality_value(parent_examples)
    if len(examples) < 5 or depth >= max_depth: # Poda
        return plurality_value(examples)
    elif all_same_classification(examples):
        return get_classification(examples[0])
    elif not attributes:
        return plurality_value(examples)
    else:
        A = max(attributes, key=lambda a: importance(a, examples))
        tree = {A: {}}
        for v in possible_values(A, examples):
            exs = [e for e in examples if get_value(e, A) == v]
            res_attrs = [attr for attr in attributes if attr != A]
            tree[A][v] = learn_decision_tree(exs, res_attrs, examples, max_depth, depth + 1)
        return tree

# --- LÓGICA DA FLORESTA ---
def treinar_random_forest(dados_treino, atributos, n_arvores=15, profundidade_maxima=10): # Adicione aqui
    floresta = []
    print(f"Construindo Floresta com {n_arvores} árvores (Profundidade: {profundidade_maxima})...")
    for i in range(n_arvores):
        amostra = random.choices(dados_treino, k=int(len(dados_treino) * 0.7))
        atrs_sorteados = random.sample(atributos, k=min(25, len(atributos)))
        
        # Repasse o valor aqui para a função learn_decision_tree
        arvore = learn_decision_tree(amostra, atrs_sorteados, [], max_depth=profundidade_maxima) 
        
        floresta.append(arvore)
        print(f"  > Árvore {i+1} pronta.")
    return floresta