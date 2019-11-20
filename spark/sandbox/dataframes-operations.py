# %%
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName ("Dataframes Operations").getOrCreate ()

# %%loading the dataset
df = spark.read.csv ("appl_stock.csv", inferSchema=True, header=True)
df.printSchema ()

# %%
x = df.filter ((df["Low"] == 197.16)).collect ()
print (x)
# %%groupby and aggregate functions
df = spark.read.csv ("sales_info.csv", inferSchema=True, header=True)
df.show ()
# %%
df.printSchema ()

# %%
group = df.groupBy ("Company").count ()
group.show ()
type (group)

# %%
df.agg ({"sales": "sum"}).show ()

# %%
from pyspark.sql.functions import countDistinct, avg, stddev

df.select (avg ("Sales").alias ("Average Sales")).show ()

# %%ordering ascending
df.orderBy ("Sales").show ()
#order descending
df.orderBy (df["Sales"].desc ()).show ()
df["Sales"].desc ()
# %%
