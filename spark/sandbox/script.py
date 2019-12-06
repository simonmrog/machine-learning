# %%
from pyspark.sql import SparkSession

# %%
spark = SparkSession.builder.appName ("LR").getOrCreate ()

# %%turning data into libsvm format
from pyspark.ml.feature import VectorAssembler

dataset = spark.read.format ("csv").options (header="true", inferSchema="true").load ("iris.csv")

assembler = VectorAssembler (inputCols=["SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"], outputCol="features")

df = assembler.transform (dataset)
df = df.select (["features", "CLASS"])
df = df.withColumnRenamed ("CLASS", "class")
df.printSchema ()