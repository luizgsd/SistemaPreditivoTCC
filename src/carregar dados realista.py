#Arquivo feito apenas para guardar o código de carregar pacotes sem roteirização, para usar depois
"""
@st.cache_data 
def carregar_dados():
    caminho_dados = os.path.join(RAIZ_PROJETO, 'data', 'processed', 'pacotes_para_apresentacao.csv')
    df = pd.read_csv(caminho_dados)
    X = df.drop(columns=['Label_Binario']).replace([np.inf, -np.inf], np.nan).fillna(0)
    y_real = df['Label_Binario'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, y_real
"""