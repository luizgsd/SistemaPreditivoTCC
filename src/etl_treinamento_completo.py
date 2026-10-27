# ESTE CÓDIGO FOI FEITO PARA SER RODADO DENTRO DO GOOGLE COLAB COM ACESSO AO GOOGLE DRIVE

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"GPU detectada? {tf.test.is_gpu_available()}")
print("="*50)

#Iniciar Spark na nuvem com mais memória
spark = SparkSession.builder \
    .appName("TCC_BigData_Colab") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#Carregar os CSVs
caminho_dados = "/content/drive/MyDrive/TCCSistemaPreditivo/data/*.csv"
print("Carregando todos os arquivos da base de dados")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(caminho_dados)

df = df.toDF(*[c.strip() for c in df.columns]) #Limpa nomes de colunas

#Limpeza ETL
print("Limpando dados e removendo colunas tendenciosas")
colunas_remover = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]
df_limpo = df.drop(*colunas_remover)

if "Flow Bytes/s" in df_limpo.columns:
    df_limpo = df_limpo.filter(
        (col("Flow Bytes/s").isNotNull()) &
        (~isnan(col("Flow Bytes/s"))) &
        (col("Flow Bytes/s") != float('inf'))
    )

df_limpo = df_limpo.withColumn("Label_Binario", when(col("Label") == "BENIGN", 0).otherwise(1)).drop("Label")

#Separa dados pra banca e salva no drive
print("Separando e guardando a amostra pra banca")
df_treino_teste, df_banca = df_limpo.randomSplit([0.995, 0.005], seed=42)
caminho_banca = "/content/drive/MyDrive/TCCSistemaPreditivo/pacotes_para_apresentacao_banca.csv"
df_banca.toPandas().to_csv(caminho_banca, index=False)
print(f"      -> Arquivo da banca guardado")

#Processamento para o TensorFlow
print("\nOtimizando o volume pra RAM do Colab")
df_treino_otimizado = df_treino_teste.sample(fraction=0.15, seed=42)

print("Convertendo a amostra otimizada")
df_modelo = df_treino_otimizado.toPandas()

print("Preparando variáveis e reduzindo peso")
X = df_modelo.drop(columns=['Label_Binario']).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
y = df_modelo['Label_Binario'].astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Deep Learning
print("\nConstruindo e treinando a rede neural com GPU")
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=512,
    validation_split=0.2,
    verbose=1
)

#Avaliação e Exportação
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print("\n" + "="*50)
print("Treinamento finalizado")
print(f"Precisão nos dados reais: {accuracy * 100:.2f}%")
print("="*50)

#Salva o arquivo de cérebro da IA no drive
caminho_modelo = "/content/drive/MyDrive/TCCSistemaPreditivo/modelo_tcc.h5"
model.save(caminho_modelo)
print(f"\nModelo de IA salvo na nuvem: {caminho_modelo}")