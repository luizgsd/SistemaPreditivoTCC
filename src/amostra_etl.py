import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan
import warnings

# Ocultar avisos do Future do terminal para deixar a tela mais limpa
warnings.filterwarnings("ignore")

print("Iniciando o motor Spark para processamento...")
spark = SparkSession.builder \
    .appName("TCC_Extracao_Amostra") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR") # Ocultar avisos do Hadoop no Windows

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
arquivo_csv = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
caminho_entrada = os.path.join(BASE_DIR, "data", "raw", arquivo_csv)
caminho_saida = os.path.join(BASE_DIR, "data", "processed", "amostra_5_percent.parquet")

print(f"\n[1/5] Carregando os dados de: {arquivo_csv}")
df = spark.read.option("header", "true").option("inferSchema", "true").csv(caminho_entrada)

# Limpar espaços invisíveis nos nomes das colunas
df = df.toDF(*[c.strip() for c in df.columns])
total_original = df.count()
print(f"      Total de pacotes originais: {total_original}")

print("\n[2/5] Removendo colunas que causam viés (IPs, Portas de Origem, etc)...")
colunas_remover = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp"]
df_limpo = df.drop(*colunas_remover)

print("\n[3/5] Tratando valores nulos e infinitos...")
if "Flow Bytes/s" in df_limpo.columns:
    df_limpo = df_limpo.filter(
        (col("Flow Bytes/s").isNotNull()) & 
        (~isnan(col("Flow Bytes/s"))) & 
        (col("Flow Bytes/s") != float('inf'))
    )

print("\n[4/5] Convertendo os Rótulos (Ataque = 1, Normal = 0)...")
df_limpo = df_limpo.withColumn(
    "Label_Binario", 
    when(col("Label") == "BENIGN", 0).otherwise(1)
).drop("Label") # Remove a coluna de texto original

print("\n[5/5] Extraindo a amostra de 5% e salvando (Método CSV Nativo)...")
df_amostra = df_limpo.sample(withReplacement=False, fraction=0.05, seed=42)
total_amostra = df_amostra.count()

# Redefinir a extensão para .csv
caminho_saida_csv = caminho_saida.replace(".parquet", ".csv")

print("      Convertendo Spark para Pandas...")
df_pandas = df_amostra.toPandas()

print("      Salvando arquivo CSV limpo...")
os.makedirs(os.path.dirname(caminho_saida_csv), exist_ok=True)
df_pandas.to_csv(caminho_saida_csv, index=False)

print("\n" + "="*50)
print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print(f"    De {total_original} pacotes, separamos {total_amostra} para a IA.")
print(f"    Arquivo salvo em: {caminho_saida_csv}")
print("="*50)