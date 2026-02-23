import math
from collections import Counter

# --- FUNÇÕES DE ACESSO E UTILIDADE ---
#ID3
def get_classification(example):
    """Extrai o valor do alvo (target) de um exemplo."""
    return example['target']

def get_value(example, attribute):
    """Retorna o valor de um atributo específico para um dado exemplo."""
    return example[attribute]

def plurality_value(examples):
    """Retorna o valor (classe) mais comum entre os exemplos fornecidos."""
    if not examples:
        return None
    labels = [get_classification(e) for e in examples] 
    contagem = Counter(labels)
    return contagem.most_common(1)[0][0]

def all_same_classification(examples):
    """Retorna True se todos os exemplos tiverem a mesma classificação."""
    if not examples:
        return True
    primeira_classe = get_classification(examples[0])
    for e in examples:
        if get_classification(e) != primeira_classe:
            return False
    return True

def possible_values(attribute, examples):
    """Retorna os valores únicos que o atributo assume no conjunto."""
    return list(set(get_value(e, attribute) for e in examples))

# --- FUNÇÕES MATEMÁTICAS (CÉREBRO) ---

def entropy(examples):
    """Calcula a Entropia de Shannon."""
    if not examples:
        return 0.0
    labels = [get_classification(e) for e in examples]
    total_examples = len(labels)
    contagem_classes = Counter(labels)
    
    ent = 0.0
    for contagem in contagem_classes.values():
        probabilidade = contagem / total_examples
        ent -= probabilidade * math.log2(probabilidade)
    return ent

def importance(attribute, examples):
    """Calcula o Ganho de Informação."""
    if not examples:
        return 0.0
    entropia_base = entropy(examples)
    total_examples = len(examples)
    
    subconjuntos = {}
    for e in examples:
        v = get_value(e, attribute)
        if v not in subconjuntos:
            subconjuntos[v] = []
        subconjuntos[v].append(e)
        
    entropia_esperada = 0.0
    for subconjunto in subconjuntos.values():
        peso = len(subconjunto) / total_examples
        entropia_esperada += peso * entropy(subconjunto)
        
    return entropia_base - entropia_esperada

# --- FUNÇÃO PRINCIPAL (ALGORITMO DO PROFESSOR) ---

# Adicione o parâmetro max_depth e depth
def learn_decision_tree(examples, attributes, parent_examples, max_depth=10, depth=0):
   # Na função learn_decision_tree, adicione:
    if not examples:
        return plurality_value(parent_examples)
    if len(examples) < 5: # Se tiver menos de 5 exemplos, não divida mais
        return plurality_value(examples)
    elif all_same_classification(examples):
        return get_classification(examples[0])
    elif not attributes or depth >= max_depth: # <--- NOVA CONDIÇÃO DE PARADA
        return plurality_value(examples)
    else:
        A = max(attributes, key=lambda a: importance(a, examples))
        tree = {A: {}}
        for v in possible_values(A, examples):
            exs = [e for e in examples if get_value(e, A) == v]
            remaining_attributes = [attr for attr in attributes if attr != A]
            
            # Passa a profundidade adiante
            subtree = learn_decision_tree(exs, remaining_attributes, examples, max_depth, depth + 1)
            tree[A][v] = subtree
        return tree