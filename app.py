import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --- IMPORTAÇÕES DO SEU PROJETO ---
# Aqui o código conecta com a lógica que você já criou (ID3 e Random Forest)
from Random_Forest_Caseiro import treinar_random_forest
from treinamento import extrair_todas_as_formas, prever_individual

# Configuração da página do navegador
st.set_page_config(page_title="IA - Random Forest Caseira", layout="wide")

st.title("Diagnóstico: Random Forest Caseira")
st.markdown("Visualizando a decisão da Floresta ID3 com importações otimizadas.")

# @st.cache_resource: Serve para o Streamlit não treinar a IA toda vez que você mexer em um botão.
# Ele guarda o modelo treinado na memória.
@st.cache_resource
def treinar_modelo_completo():
    digits = load_digits()
    
    # Binarização: Transforma tons de cinza em 0 (preto) ou 1 (branco)
    # Isso facilita a decisão do algoritmo ID3 (baseado em Sim/Não)
    X_bin = (digits.data > 8).astype(int) 
    y = digits.target
    
    examples = []
    for i in range(len(X_bin)):
        # Cria um dicionário para cada imagem onde cada pixel é um atributo (px_0, px_1...)
        ex = {f"px_{j}": X_bin[i][j] for j in range(64)}
        # Adiciona características extras (formas) extraídas da sua função personalizada
        ex.update(extrair_todas_as_formas(X_bin[i])) 
        ex['target'] = y[i]
        examples.append(ex)

    # Define a lista de nomes de todos os atributos que a IA pode usar
    atrs = [f"px_{i}" for i in range(64)] + ['f_reta_topo', 'f_curva_base', 'f_muitos_cruzamentos', 'f_centro_preenchido']
    
    # Separa 80% para treinar a IA e 20% para testar se ela aprendeu mesmo
    train, test = train_test_split(examples, test_size=0.2, random_state=42)

    # Chama a sua função de treino para criar a "Floresta" (conjunto de árvores ID3)
    floresta = treinar_random_forest(train, atrs, n_arvores=100, profundidade_maxima=12)
    
    return floresta, test

# Executa o treino e guarda os resultados
floresta, test_examples = treinar_modelo_completo()

# --- SIDEBAR (Barra Lateral) ---
# Cria um slider para você escolher qual imagem do teste quer analisar
idx = st.sidebar.slider("Selecione um exemplo do teste:", 0, len(test_examples)-1, 15)
exemplo_selecionado = test_examples[idx]

# --- ANÁLISE DE IMPORTÂNCIA ---
def calcular_importancia_pixel(floresta):
    """
    Esta função percorre todas as árvores da floresta e conta 
    quantas vezes cada pixel (atributo) foi usado para tomar uma decisão.
    """
    contagem = Counter()
    for arvore in floresta:
        def percorrer(no):
            if isinstance(no, dict): # Se o nó não é uma folha (é uma decisão)
                atrib = list(no.keys())[0]
                contagem[atrib] += 1 # Conta o atributo usado
                # Continua descendo na árvore recursivamente
                for sub in no[atrib].values(): percorrer(sub)
        percorrer(arvore)
    
    # Organiza a contagem em uma matriz 8x8 (formato da imagem)
    imp_matrix = np.zeros(64)
    for i in range(64):
        imp_matrix[i] = contagem.get(f"px_{i}", 0)
    return imp_matrix.reshape(8, 8)

# --- INTERFACE VISUAL (Colunas) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Visualização dos Pixels")
    fig, ax = plt.subplots()
    
    # Desenha a imagem do dígito selecionado
    img_data = np.array([exemplo_selecionado[f"px_{i}"] for i in range(64)]).reshape(8, 8)
    ax.imshow(img_data, cmap='gray_r')
    
    # Sobrepõe um "mapa de calor" vermelho mostrando onde a IA mais "olha"
    importancia = calcular_importancia_pixel(floresta)
    ax.imshow(importancia, cmap='Reds', alpha=0.4) # alpha 0.4 deixa transparente
    ax.axis('off')
    st.pyplot(fig)
    st.info("O tom vermelho indica os pixels mais consultados pela floresta.")

with col2:
    st.subheader("Resultado da Votação")
    
    # Faz cada árvore da floresta dar o seu "palpite" individual
    votos = [prever_individual(arv, exemplo_selecionado) for arv in floresta]
    votos = [v for v in votos if v is not None] # Remove votos nulos
    
    # Contabiliza qual número recebeu mais votos
    contagem_votos = Counter(votos)
    previsao = contagem_votos.most_common(1)[0][0] if votos else 0
    real = exemplo_selecionado['target']
    confianca = (contagem_votos[previsao] / len(votos)) * 100 if votos else 0

    # Mostra o resultado principal com uma métrica visual
    st.metric("Predição da Floresta", f"Dígito {previsao}", delta=f"{confianca:.1f}% de votos")
    
    # Feedback de Acerto ou Erro
    if previsao == real:
        st.success(f"Acerto! O número real é {real}")
    else:
        st.error(f"Erro! O número real é {real}")

    # Gráfico de barras mostrando a distribuição dos votos para cada número (0-9)
    df_votos = pd.DataFrame({
        "Dígito": list(range(10)),
        "Votos": [contagem_votos.get(i, 0) for i in range(10)]
    })
    st.bar_chart(df_votos.set_index("Dígito"))

# Mostra as características extras (como 'f_curva_base') que foram extraídas
with st.expander("Ver Atributos de Forma extraídos"):
    formas = {k: exemplo_selecionado[k] for k in exemplo_selecionado if k.startswith('f_')}
    st.json(formas)
