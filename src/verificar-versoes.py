from pyspark.sql import SparkSession
import pyspark
import sys
import numpy as np
import scipy
import tensorflow as tf
import findspark

findspark.init()

print("="*40)
print("DIAGNÓSTICO DO AMBIENTE DO TCC")
print("="*40)

print(f"Python     : {sys.version.split(' ')[0]}")
print(f"NumPy      : {np.__version__}")
print(f"SciPy      : {scipy.__version__}")
print(f"TensorFlow : {tf.__version__}")
print(f"PySpark    : {pyspark.__version__}")

try:
    spark = SparkSession.builder.appName(
        "CheckEnv").master("local[*]").getOrCreate()
    print(f"Motor Spark: Rodando perfeitamente (Versão {spark.version})")
    spark.stop()
except Exception as e:
    print("\nErro ao iniciar o motor do Spark (possível problema com o Java):")
    print(e)

print("="*40)
