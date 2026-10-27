import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
import warnings
warnings.filterwarnings("ignore")

print("Iniciando o Spark")
spark = SparkSession.builder \
    .appName("TCC_Extracao_Amostra") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
arquivo_csv = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
caminho_entrada = os.path.join(BASE_DIR, "data", "raw", arquivo_csv)
caminho_saida = os.path.join(BASE_DIR, "data", "processed", "amostra_5_percent.parquet")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(caminho_entrada)

#Limpar espaços invisíveis nos nomes das colunas
df = df.toDF(*[c.strip() for c in df.columns])
total_original = df.count()
print(f"Total de pacotes originais: {total_original}")

print("\nRemovendo colunas que causam viés")
colunas_remover = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]
df_limpo = df.drop(*colunas_remover)

print("\nTratando valores nulos e infinitos")
if "Flow Bytes/s" in df_limpo.columns:
    df_limpo = df_limpo.filter(
        (col("Flow Bytes/s").isNotNull()) & 
        (~isnan(col("Flow Bytes/s"))) & 
        (col("Flow Bytes/s") != float('inf'))
    )

print("\nConvertendo os rótulos (Ataque = 1, Normal = 0)")
df_limpo = df_limpo.withColumn(
    "Label_Binario", 
    when(col("Label") == "BENIGN", 0).otherwise(1)
).drop("Label")

print("\nExtraindo a amostra de 5% e salvando")
df_amostra = df_limpo.sample(withReplacement=False, fraction=0.05, seed=42)
total_amostra = df_amostra.count()

caminho_saida_csv = caminho_saida.replace(".parquet", ".csv")

print("Convertendo Spark para Pandas")
df_pandas = df_amostra.toPandas()

print("Salvando arquivo CSV limpo")
os.makedirs(os.path.dirname(caminho_saida_csv), exist_ok=True)
df_pandas.to_csv(caminho_saida_csv, index=False)

print("\n" + "="*50)
print("Processamento concluido")
print(f"De {total_original} pacotes, foram separados {total_amostra} para a IA.")
print(f"Arquivo salvo em: {caminho_saida_csv}")
print("="*50)