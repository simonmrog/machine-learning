# %% importing the libraries
from pyspark.sql import SparkSession

# %% starting spark session
spark = SparkSession.builder.appName ("Linear Regression").getOrCreate ()

# %% loading the dataset
data = spark.read.csv ("ecommerce-customers.csv", inferSchema=True, header=True)
data.printSchema ()

# %% setting dataset for machine learning
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler (inputCols=["Avg Session Length", "Time on App", "Length of Membership", "Time on Website"], outputCol="features")

df = assembler.transform (data)
df = df.select ("features", "Yearly Amount Spent")
df.show ()

# %%train-test splitting
train, test = df.randomSplit ([0.7, 0.3])
test.describe ().show ()

# %%fitting the model to the data
from pyspark.ml.regression import LinearRegression
lr = LinearRegression (featuresCol="features", labelCol="Yearly Amount Spent")
print (type (lr))
model = lr.fit (train)
print (type (model))

# %%evaluating the model
test_results = model.evaluate (test)
test_results.residuals.show ()
type (test_results)

# %%root mean squared error
test_results.rootMeanSquaredError

# %%R2 coefficient
test_results.r2

# %%predicting values never seen by the model
utest = test.select ("features")
pred = model.transform (utest)
pred.show ()
# %%
test.show ()

# %%
