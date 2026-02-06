import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IA - Analise de Decisao", layout="wide")

st.title("Diagnostico de Decisao da IA")
st.markdown("---")

@st.cache_resource
def treinar_modelo():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    modelo = DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo, digits, X_test, y_test

modelo, digits, X_test, y_test = treinar_modelo()

st.sidebar.header("Configuracoes")
idx_exemplo = st.sidebar.slider("Escolha uma imagem de teste:", 0, len(X_test)-1, 15)

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Mapa de Analise")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(X_test[idx_exemplo].reshape(8, 8), cmap='gray_r', interpolation='nearest')
    
    indicador = modelo.decision_path([X_test[idx_exemplo]])
    nos_percorridos = indicador.indices[indicador.indptr[0]:indicador.indptr[1]]
    
    detalhes_decisao = []
    contador = 1
    
    for no in nos_percorridos:
        if modelo.tree_.feature[no] != -2:
            pixel = modelo.tree_.feature[no]
            valor = X_test[idx_exemplo][pixel]
            corte = modelo.tree_.threshold[no]
            
            row, col = divmod(pixel, 8)

            ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, fill=False, color='red', lw=2))
            ax.text(col, row, str(contador), color='white', fontsize=12, 
                    fontweight='bold', ha='center', va='center', 
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
            
            status = "Escuro (Tinta)" if valor > corte else "Claro (Papel)"
            explificacao = f"IA buscou {status}"
            
            detalhes_decisao.append({
                "Ordem": f"{contador}º Passo",
                "Localizacao (Pixel)": pixel,
                "O que a IA viu?": status,
                "Forca do Traco": f"{int((valor/16)*100)}%"
            })
            contador += 1
            
    ax.axis('off')
    st.pyplot(fig)
    st.info("A IA analisa os pontos numerados acima na ordem indicada para chegar ao veredito.")

with col2:
    st.subheader("Veredito da Inteligencia Artificial")
    previsao = modelo.predict([X_test[idx_exemplo]])[0]
    real = y_test[idx_exemplo]
    
    st.metric(label="Numero Identificado", value=f"Dígito {previsao}", delta="Sucesso" if previsao == real else "Erro")
    
    if previsao != real:
        st.error(f"Nota: O numero real esperado era {real}.")

    st.write("---")
    st.write("**Linha de Raciocinio:**")
    
    if detalhes_decisao:
        df = pd.DataFrame(detalhes_decisao)
        st.table(df.set_index('Ordem'))
    
    with st.expander("O que sao esses passos?"):
        st.write("""
        Para cada ponto (pixel) numerado na imagem:
        1. A IA mede a **Forca do Traço** (0% é branco total, 100% é preto total).
        2. Ela compara com uma **Regra de Corte** aprendida no treino.
        3. Se o ponto for mais escuro que o limite, ela segue por um caminho da arvore; se for mais claro, segue por outro.
        """)

st.markdown("---")
st.caption("Desenvolvido para fins academicos - Algoritmo Decision Tree Classifier.")