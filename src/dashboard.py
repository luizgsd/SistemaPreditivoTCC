import os
import streamlit as st
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import random
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard IDS - TCC", page_icon="🛡️", layout="wide")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROJETO = os.path.dirname(DIRETORIO_ATUAL)

#Carregamento do modelo
@st.cache_resource 
def carregar_modelo(tamanho_entrada):
    tf.keras.backend.clear_session()
    
    #Cria e fixa uma sessão
    sessao = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sessao)
    
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(tamanho_entrada,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    caminho_modelo = os.path.join(RAIZ_PROJETO, 'modelo_tcc.h5')
    modelo.load_weights(caminho_modelo)
    
    grafo = tf.compat.v1.get_default_graph()
    
    return modelo, grafo, sessao

@st.cache_data 
def carregar_dados():
    caminho_dados = os.path.join(RAIZ_PROJETO, 'data', 'processed', 'pacotes_para_apresentacao.csv')
    df = pd.read_csv(caminho_dados)
    X = df.drop(columns=['Label_Binario']).replace([np.inf, -np.inf], np.nan).fillna(0)
    y_real = df['Label_Binario'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, y_real

if 'monitorando' not in st.session_state:
    st.session_state.monitorando = False
    st.session_state.indice_atual = 0
    st.session_state.total_pacotes = 0
    st.session_state.total_normais = 0
    st.session_state.total_ataques = 0
    st.session_state.historico = pd.DataFrame()
    st.session_state.historico_prob = []
    st.session_state.alertas = []

def alternar_monitoramento():
    st.session_state.monitorando = not st.session_state.monitorando

#Interface Lateral
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2885/2885412.png", width=100)
st.sidebar.title("Configurações do IDS")
st.sidebar.markdown("Sistema Preditivo de Ataques Cibernéticos utilizando Deep Learning.")
st.sidebar.markdown("---")

velocidade = st.sidebar.slider("Velocidade da Simulação (Segundos)", 0.1, 2.0, 0.5)

rotulo_botao = "PARAR MONITORAMENTO" if st.session_state.monitorando else "INICIAR MONITORAMENTO"
st.sidebar.button(rotulo_botao, on_click=alternar_monitoramento)
espaco_botao_zerar = st.sidebar.empty()

if not st.session_state.monitorando and st.session_state.indice_atual > 0:
    if espaco_botao_zerar.button("ZERAR SIMULAÇÃO"):
        st.session_state.monitorando = False
        st.session_state.indice_atual = 0
        st.session_state.total_pacotes = 0
        st.session_state.total_normais = 0
        st.session_state.total_ataques = 0
        st.session_state.historico = pd.DataFrame()
        st.session_state.historico_prob = []
        st.session_state.alertas = []
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()
else:
    espaco_botao_zerar.empty() 

#Tela Principal
st.title("Centro de Operações de Segurança")
st.markdown("Monitoramento de Tráfego de Rede em Tempo Real")
st.markdown("""
<style>
.caixa-alertas {
    position: fixed;
    top: 80px;
    right: 20px;
    z-index: 999999;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.alerta-item {
    background-color: rgba(239, 85, 59, 0.95);
    color: white;
    padding: 15px 25px;
    border-radius: 8px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    border-left: 6px solid #8b0000;
    font-family: sans-serif;
    font-size: 14px;
    animation: fadeIn 0.4s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateX(50px); }
    to { opacity: 1; transform: translateX(0); }
}
</style>
""", unsafe_allow_html=True)

espaco_notificacoes = st.empty()

col1, col2, col3 = st.columns(3)
kpi_pacotes = col1.empty()
kpi_normais = col2.empty()
kpi_ataques = col3.empty()

st.markdown("---")
alerta_espaco = st.empty() 

col_graf1, col_graf2 = st.columns(2)
grafico_linha = col_graf1.empty()
grafico_rosca = col_graf2.empty()

st.markdown("---")
painel_status = st.empty()
titulo_historico = st.empty()
tabela_espaco = st.empty()

#Motor da Simulação
if st.session_state.monitorando:
    df_original, X_scaled, y_real = carregar_dados()
    modelo, grafo, sessao = carregar_modelo(X_scaled.shape[1])
    alerta_espaco.info("Conectado ao Switch Core. Analisando pacotes de rede.")
    titulo_historico.subheader("Histórico de Pacotes")

    while st.session_state.indice_atual < len(X_scaled):
        
        if not st.session_state.monitorando:
            alerta_espaco.warning("Monitoramento Interrompido pelo Usuário.")
            break

        i = st.session_state.indice_atual
        pacote_atual = X_scaled[i:i+1]
        dados_originais = df_original.iloc[[i]].copy()
        
        with grafo.as_default():
            tf.compat.v1.keras.backend.set_session(sessao)
            probabilidade = modelo.predict(pacote_atual)[0][0]
            
        previsao = 1 if probabilidade > 0.5 else 0
        st.session_state.historico_prob.append(probabilidade * 100)
        
        st.session_state.total_pacotes += 1
        if previsao == 0:
            st.session_state.total_normais += 1
            status = "NORMAL"
        else:
            st.session_state.total_ataques += 1
            status = "ATAQUE DETECTADO"
            
            #Tenta puxar a porta real do CSV e se não achar sorteia uma comum
            if 'Destination Port' in dados_originais.columns:
                porta_alvo = int(dados_originais['Destination Port'].values[0])
            else:
                porta_alvo = random.choice([80, 443, 22, 21, 3389])
                
            #Gera IPs simulados pra origem e destino
            ip_externo = f"{random.randint(11, 220)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
            ip_interno = f"10.0.0.{random.randint(10, 250)}"
            
            #Notificação de ataque detectado
            mensagem_html = (
                f"<b>AMEAÇA INTERCEPTADA</b><br>"
                f"<span style='font-size: 12px;'>"
                f"<b>Origem:</b> {ip_externo} &rarr; <b>Alvo:</b> {ip_interno}:{porta_alvo}<br>"
                f"<b>Confiança IA:</b> {probabilidade*100:.2f}% | <b>Ação:</b> IP Bloqueado"
                f"</span>"
            )
            
            st.session_state.alertas.append({
                'msg': mensagem_html, 
                'time': time.time()
            })
            
        dados_originais['Status IDS'] = status
        dados_originais['Confiança IA'] = f"{probabilidade*100:.2f}%"
        
        st.session_state.historico = pd.concat([dados_originais, st.session_state.historico]).head(8)
        
        kpi_pacotes.metric("Pacotes Analisados", st.session_state.total_pacotes)
        kpi_normais.metric("Tráfego Normal", st.session_state.total_normais)
        kpi_ataques.metric("Ameaças Bloqueadas", st.session_state.total_ataques)
        
        fig_linha = go.Figure(data=go.Scatter(y=st.session_state.historico_prob[-40:], mode='lines+markers', line=dict(color='#00F0FF')))
        fig_linha.update_layout(title="Grau de Ameaça em Tempo Real (%)", height=300, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_linha.update_yaxes(range=[0, 100])
        grafico_linha.plotly_chart(fig_linha, use_container_width=True)

        cores = ['#00CC96', '#EF553B'] if st.session_state.total_ataques > 0 else ['#00CC96']
        fig_rosca = go.Figure(data=[go.Pie(labels=['Normal', 'Ataque'], values=[st.session_state.total_normais, st.session_state.total_ataques], hole=.6, marker_colors=cores)])
        fig_rosca.update_layout(
            title="Distribuição do Tráfego", 
            height=300, 
            margin=dict(l=20, r=20, t=40, b=20), 
            paper_bgcolor="rgba(0,0,0,0)", 
            showlegend=True, 
            legend=dict(
                orientation="v",
                yanchor="bottom", y=0.05, 
                xanchor="right", x=0.95,
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(size=15, color="white")
            )
        )
        grafico_rosca.plotly_chart(fig_rosca, use_container_width=True)

        colunas_display = list(st.session_state.historico.columns[:3]) + ['Confiança IA', 'Status IDS']
        tabela_espaco.dataframe(st.session_state.historico[colunas_display], use_container_width=True)

        #Popup de alerta
        tempo_agora = time.time()
        st.session_state.alertas = [a for a in st.session_state.alertas if tempo_agora - a['time'] < 5.0]
        
        if len(st.session_state.alertas) > 0:
            html_alertas = "<div class='caixa-alertas'>"
            for alerta in st.session_state.alertas:
                html_alertas += f"<div class='alerta-item'>{alerta['msg']}</div>"
            html_alertas += "</div>"
            
            espaco_notificacoes.markdown(html_alertas, unsafe_allow_html=True)
        else:
            espaco_notificacoes.empty()
        
        st.session_state.indice_atual += 1
        time.sleep(velocidade)
        
elif st.session_state.indice_atual > 0:
    #Tela de sistema pausado
    kpi_pacotes.metric("Pacotes Analisados", st.session_state.total_pacotes)
    kpi_normais.metric("Tráfego Normal", st.session_state.total_normais)
    kpi_ataques.metric("Ameaças Bloqueadas", st.session_state.total_ataques)
    alerta_espaco.empty()
    painel_status.info("**Inspeção de Tráfego Suspensa.** \n\nO Detector foi pausado pelo operador. Dados da sessão estão preservados em memória.\n\n* Clique em **INICIAR MONITORAMENTO** no painel lateral para retomar a análise a partir do último pacote verificado.\n* Clique em **ZERAR SIMULAÇÃO** para limpar a memória e reiniciar a operação.")
    titulo_historico.subheader("Histórico de Pacotes (Modo Leitura)")
    colunas_display = list(st.session_state.historico.columns[:3]) + ['Confiança IA', 'Status IDS']
    tabela_espaco.dataframe(st.session_state.historico[colunas_display], use_container_width=True)

else:
    #Tela inicial do sistema
    painel_status.success("**Módulo de Detecção Preditiva Pronto.** \n\nDetector inicializado com sucesso. O sistema encontra-se em modo de espera.\n\nClique em **INICIAR MONITORAMENTO** no menu lateral para iniciar a análise dos pacotes em tempo real.")