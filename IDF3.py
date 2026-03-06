import math
from collections import Counter

# --- FUNÇÕES DE ACESSO E UTILIDADE ---
# Estas funções ajudam a manter o código limpo, lidando com os dicionários de dados.

def get_classification(example):
    """Extrai o valor do alvo (target/label) de um exemplo."""
    return example['target']

def get_value(example, attribute):
    """Busca o valor de um atributo (ex: se o px_34 é 0 ou 1)."""
    return example[attribute]

def plurality_value(examples):
    """
    VOTAÇÃO DE MAIORIA: Se o algoritmo precisa parar, ele escolhe o número
    que mais aparece naquele conjunto. Se empatar, pega o primeiro.
    """
    if not examples:
        return None
    labels = [get_classification(e) for e in examples] 
    contagem = Counter(labels)
    return contagem.most_common(1)[0][0]

def all_same_classification(examples):
    """
    PUREZA: Verifica se todos os exemplos são do mesmo número (ex: tudo '8').
    Se sim, a árvore não precisa mais ser dividida aqui.
    """
    if not examples:
        return True
    primeira_classe = get_classification(examples[0])
    for e in examples:
        if get_classification(e) != primeira_classe:
            return False
    return True

def possible_values(attribute, examples):
    """Identifica quais caminhos o atributo pode seguir (geralmente 0 ou 1)."""
    return list(set(get_value(e, attribute) for e in examples))

# --- FUNÇÕES MATEMÁTICAS (CÉREBRO) ---

def entropy(examples):
    """
    ENTROPIA DE SHANNON: Mede a desordem ou incerteza.
    Fórmula: H(S) = - ∑ p_i * log2(p_i)
    Se todos forem do mesmo número, a entropia é 0 (ordem total).
    """
    if not examples:
        return 0.0
    labels = [get_classification(e) for e in examples]
    total_examples = len(labels)
    contagem_classes = Counter(labels)
    
    ent = 0.0
    for contagem in contagem_classes.values():
        probabilidade = contagem / total_examples
        # O log2 mede a 'surpresa' da informação
        ent -= probabilidade * math.log2(probabilidade)
    return ent

def importance(attribute, examples):
    """
    GANHO DE INFORMAÇÃO: É a 'utilidade' de um atributo.
    Calcula quanto a entropia diminui se dividirmos os dados por este atributo.
    Atributos que separam bem os números ganham notas maiores.
    """
    if not examples:
        return 0.0
    entropia_base = entropy(examples)
    total_examples = len(examples)
    
    # Agrupa exemplos pelos valores do atributo (0 ou 1)
    subconjuntos = {}
    for e in examples:
        v = get_value(e, attribute)
        if v not in subconjuntos:
            subconjuntos[v] = []
        subconjuntos[v].append(e)
        
    # Calcula a média ponderada da entropia dos subconjuntos
    entropia_esperada = 0.0
    for subconjunto in subconjuntos.values():
        peso = len(subconjunto) / total_examples
        entropia_esperada += peso * entropy(subconjunto)
        
    # O ganho é a diferença entre a dúvida inicial e a dúvida após o teste
    return entropia_base - entropia_esperada

# --- FUNÇÃO PRINCIPAL (ALGORITMO ID3 RECURSIVO) ---

def learn_decision_tree(examples, attributes, parent_examples, max_depth=10, depth=0):
    """
    CONSTRUÇÃO DA ÁRVORE: Implementação do pseudocódigo do livro de Russell & Norvig.
    Cria uma estrutura de dicionários aninhados que representa a árvore.
    """
    # CASO BASE 1: Se não há mais exemplos, usa a maioria do pai (evita galhos vazios)
    if not examples:
        return plurality_value(parent_examples)
    
    # CASO BASE 2: Pré-poda (Pre-pruning). Se o grupo é muito pequeno, para aqui.
    if len(examples) < 5: 
        return plurality_value(examples)
    
    # CASO BASE 3: Se todos os exemplos são da mesma classe, virou uma folha.
    elif all_same_classification(examples):
        return get_classification(examples[0])
    
    # CASO BASE 4: Se não há mais atributos para testar ou atingiu o limite de altura.
    elif not attributes or depth >= max_depth:
        return plurality_value(examples)
    
    else:
        # ESCOLHA DO LÍDER: Seleciona o atributo com o maior Ganho de Informação
        A = max(attributes, key=lambda a: importance(a, examples))
        
        # Cria o nó da árvore (um dicionário onde a chave é o nome do atributo)
        tree = {A: {}}
        
        # Para cada valor possível deste atributo (ex: 0 e 1), cria um sub-galho
        for v in possible_values(A, examples):
            exs = [e for e in examples if get_value(e, A) == v]
            # Remove o atributo já usado para não repetir o teste no próximo nível
            remaining_attributes = [attr for attr in attributes if attr != A]
            
            # RECURSIVIDADE: Chama a função para construir o próximo nível
            subtree = learn_decision_tree(exs, remaining_attributes, examples, max_depth, depth + 1)
            tree[A][v] = subtree
            
        return tree
