import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IA - Analise de Decisao (Ensemble)", layout="wide")

st.title("Diagnóstico de Decisão da IA (Random Forest)")
st.markdown("---")

@st.cache_resource
def treinar_modelo():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42, stratify=digits.target
    )
    modelo = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, digits, X_test, y_test

modelo, digits, X_test, y_test = treinar_modelo()

st.sidebar.header("Configurações")
idx_exemplo = st.sidebar.slider("Escolha uma imagem de teste:", 0, len(X_test)-1, 15)

col1, col2 = st.columns([1, 1.3])

arvore_exemplo = modelo.estimators_[0]

with col1:
    st.subheader("Mapa de Análise (Caminho de Decisão)")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(X_test[idx_exemplo].reshape(8, 8), cmap='gray_r', interpolation='nearest')

    indicador = arvore_exemplo.decision_path([X_test[idx_exemplo]])
    nos_percorridos = indicador.indices
    
    detalhes_decisao = []
    contador = 1
    
    for no in nos_percorridos:
        if arvore_exemplo.tree_.feature[no] != -2:
            pixel = arvore_exemplo.tree_.feature[no]
            valor = X_test[idx_exemplo][pixel]
            corte = arvore_exemplo.tree_.threshold[no]
            
            row, col = divmod(pixel, 8)

            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='red', lw=2))
            ax.text(col, row, str(contador), color='white', fontsize=12, 
                    fontweight='bold', ha='center', va='center', 
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
            
            status = "Escuro (Tinta)" if valor > corte else "Claro (Papel)"
            
            detalhes_decisao.append({
                "Ordem": f"{contador}º Passo",
                "Localização (Pixel)": f"Linha {row}, Col {col}",
                "O que a IA viu?": status,
                "Intensidade": f"{int((valor/16)*100)}%"
            })
            contador += 1
            
    ax.axis('off')
    st.pyplot(fig)
    st.info("A numeração indica a sequência lógica que uma das árvores da floresta seguiu.")

with col2:
    st.subheader("Veredito da Inteligência Artificial")
    previsao = modelo.predict([X_test[idx_exemplo]])[0]
    real = y_test[idx_exemplo]
    
    probabilidades = modelo.predict_proba([X_test[idx_exemplo]])[0]
    confianca = probabilidades[previsao] * 100

    st.metric(label="Número Identificado", value=f"Dígito {previsao}", 
              delta=f"{confianca:.1f}% de Certeza", delta_color="normal" if previsao == real else "inverse")
    
    if previsao != real:
        st.error(f"Nota: O número real esperado era {real}.")
    else:
        st.success(f"A IA acertou! O dígito real é {real}.")

    st.write("---")
    st.write("**Raciocínio Detalhado (Exemplo de uma Árvore):**")
    
    if detalhes_decisao:
        df = pd.DataFrame(detalhes_decisao)
        st.table(df.set_index('Ordem'))
    
    with st.expander("Como a IA calcula essa Certeza?", expanded=False):
        st.write(f"""
        Esta Inteligência Artificial é uma **Random Forest** (Floresta Aleatória), composta por **100 árvores de decisão** independentes.
        
        **O cálculo funciona assim:**
        1. Cada uma das 100 árvores analisa a imagem e dá o seu próprio palpite.
        2. No exemplo atual, **{int(confianca)} árvores** votaram no dígito {previsao}.
        3. A "Certeza" de {confianca:.1f}% é exatamente a proporção de árvores que concordaram com esse veredito.
        
        Isso torna o sistema muito mais robusto, pois evita que o erro de uma única árvore comprometa o resultado final.
        """)



st.markdown("---")
st.caption(" Implementação de Ensemble Learning (Random Forest) para Reconhecimento de Dígitos.")