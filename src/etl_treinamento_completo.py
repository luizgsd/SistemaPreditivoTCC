# ESTE CÓDIGO FOI FEITO PARA SER RODADO DENTRO DO GOOGLE COLAB COM ACESSO AO GOOGLE DRIVE

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"Placa de Vídeo (GPU) detectada? {tf.test.is_gpu_available()}")
print("="*50)

# 1. Iniciar Spark na Nuvem com mais memória
spark = SparkSession.builder \
    .appName("TCC_BigData_Colab") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# 2. Carregar TODOS os CSVs de uma vez só (A mágica do Big Data)
caminho_dados = "/content/drive/MyDrive/TCCSistemaPreditivo/data/*.csv"
print("[1/5] Carregando todos os arquivos da base de dados...")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(caminho_dados)

# Limpar nomes de colunas
df = df.toDF(*[c.strip() for c in df.columns])

# 3. Limpeza (Pipeline de ETL)
print("[2/5] Limpando dados e removendo colunas tendenciosas...")
colunas_remover = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]
df_limpo = df.drop(*colunas_remover)

if "Flow Bytes/s" in df_limpo.columns:
    df_limpo = df_limpo.filter(
        (col("Flow Bytes/s").isNotNull()) &
        (~isnan(col("Flow Bytes/s"))) &
        (col("Flow Bytes/s") != float('inf'))
    )

df_limpo = df_limpo.withColumn("Label_Binario", when(col("Label") == "BENIGN", 0).otherwise(1)).drop("Label")

# 4. O SEGREDO DO TCC: Separar dados inéditos para a Banca (0.5% = ~14.000 pacotes)
print("[3/5] Separando e guardando a amostra secreta para a apresentação da Banca...")
df_treino_teste, df_banca = df_limpo.randomSplit([0.995, 0.005], seed=42)

# Salvar arquivo da banca direto no seu Drive
caminho_banca = "/content/drive/MyDrive/TCCSistemaPreditivo/pacotes_para_apresentacao_banca.csv"
df_banca.toPandas().to_csv(caminho_banca, index=False)
print(f"      -> Arquivo da banca guardado a 7 chaves no seu Drive!")

# 5. Processamento para o TensorFlow
print("\n[4/5] Otimizando o volume para a RAM do Colab (Amostragem Global)...")
# Puxamos 15% de toda a base (ainda serão centenas de milhares de linhas misturadas de todos os dias)
df_treino_otimizado = df_treino_teste.sample(fraction=0.15, seed=42)

print("      Convertendo a amostra otimizada. Isso vai levar um tempinho...")
df_modelo = df_treino_otimizado.toPandas()

print("      Preparando variáveis e reduzindo peso matemático (float32)...")
# O astype(np.float32) corta o consumo de memória RAM pela metade instantaneamente!
X = df_modelo.drop(columns=['Label_Binario']).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
y = df_modelo['Label_Binario'].astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Deep Learning no Colab
print("\n[5/5] Construindo e treinando a Rede Neural com GPU...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Lote maior (batch_size=512) para a GPU devorar os dados mais rápido
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=512,
    validation_split=0.2,
    verbose=1
)

# 7. Avaliação e Exportação
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print("\n" + "="*50)
print(" ✅ TREINAMENTO FINALIZADO COM SUCESSO!")
print(f"    Acurácia nos dados reais: {accuracy * 100:.2f}%")
print("="*50)

# Salvar o cérebro da IA no seu Google Drive!
caminho_modelo = "/content/drive/MyDrive/TCCSistemaPreditivo/modelo_tcc.h5"
model.save(caminho_modelo)
print(f"\n🧠 Modelo de IA salvo na nuvem: {caminho_modelo}")
print("Você já pode baixar esse modelo para usar no seu Dashboard Front-end!")