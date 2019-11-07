# %%importing the libraries
from pyspark.sql import SparkSession

# %% creating spark session
spark = SparkSession.builder.appName ("Basics").getOrCreate ()

# %%loading data
df = spark.read.json ("people.json")

# %%
df.show ()

# %%
df.printSchema ()

# %%
df.columns

# %%
df.describe ().show ()

# %%creating schema manually and loading data
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

data_schema = [StructField ("age", IntegerType (), True), StructField ("name", StringType (), True)]

final_schema = StructType (fields=data_schema)

df = spark.read.json ("people.json", schema=final_schema)
df.printSchema ()
# %%

# %%
