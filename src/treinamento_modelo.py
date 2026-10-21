import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*50)
print(f"Iniciando TensorFlow (Versão: {tf.__version__})")
print("="*50)

#Definir caminhos dinâmicos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_csv = os.path.join(BASE_DIR, "data", "processed", "amostra_5_percent.csv")

#Carregar os dados
print(f"\nCarregando a amostra processada")
df = pd.read_csv(caminho_csv)

#Pré-processamento
print("Separando dados de treino/teste e normalizando")
X = df.drop(columns=['Label_Binario'])
y = df['Label_Binario']

X = X.replace([np.inf, -np.inf], np.nan).fillna(0) #Preencher valores nulos ou infinitos remanescentes com 0 por segurança
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Dividir os dados em 80% para Treino e 20% para Teste

#Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Arquitetura da Rede Neural
print("Construindo a arquitetura da Rede Neural")
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)), #Camada de entrada + 1ª Camada oculta com 64 neurônios
    tf.keras.layers.Dropout(0.3), # Dropout
    tf.keras.layers.Dense(32, activation='relu'), #2ª Camada oculta com 32 neurônios
    tf.keras.layers.Dense(1, activation='sigmoid') #Camada de saída com 1 neurônio (0 a 1 indicando a probabilidade de ser ataque)
])

#Compilar o modelo
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

#Treinamento
print("\nIniciando o treinamento do modelo")
history = model.fit(
    X_train_scaled, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2, # Usa 20% do treino como contra-prova a cada época
    verbose=1
)

#Avaliação Final
print("\n" + "="*50)
print("Testando com dados ineditos")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"    Precisão: {accuracy * 100:.2f}%")
print(f"    Margem de Erro: {loss:.4f}")
print("="*50)