# %%Start a simple Spark Session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName ("spark").getOrCreate ()

# %%Load the Walmart Stock CSV File, have Spark infer the data types.
df = spark.read.csv ("walmart_stock.csv", inferSchema=True, header=True)

# %%What are the column names?
df.columns

# %%What does the Schema look like?
df.printSchema ()

# %%Print out the first 5 columns.
df.head (5)

# %%Use describe() to learn about the DataFrame
df.describe ().show ()

# %%There are too many decimal places for mean and stddev in the describe() dataframe. Format the numbers to just show up to two decimal places.
describe = df.describe ()
describe.printSchema ()


# %%
from pyspark.sql.functions import format_number

describe_float = describe.select (describe["summary"],
                                  format_number (describe.Open.cast ("float"), 2).alias ("Open"),
                                  format_number (describe.High.cast ("float"), 2).alias ("High"),
                                  format_number (describe.Low.cast ("float"), 2).alias ("Low"),
                                  format_number (describe.Close.cast ("float"), 2).alias ("Close"),
                                  describe.Volume.cast ("int").alias ("Volume"))
describe_float.show ()

# %%Create a new dataframe with a column called HV Ratio that is the ratio of the High Price versus volume of stock traded for a day.
df2 = df.withColumn ("HV Ratio", df.High / df.Volume)
df2.select ("HV Ratio").show ()

# %%What day had the Peak High in Price?
df.orderBy (df["High"].desc ()).head (1)[0][0]

# %%What is the mean of the Close column?
from pyspark.sql.functions import mean
df.select (mean("Close")).show ()

# %%What is the max and min of the Volume column?
from pyspark.sql.functions import max, min
df.select (max ("Volume"), min ("Volume")).show ()

# %%How many days was the Close lower than 60 dollars?
df.filter (df["Close"] < 60).count ()

# %%What percentage of the time was the High greater than 80 dollars ?
days_high = df.filter (df["High"] > 80).count ()
total = df.count ()
print (days_high / total * 100)

# %%What is the Pearson correlation between High and Volume?
from pyspark.sql.functions import corr
df.select (corr ("High", "Volume")).show ()

# %%What is the max High per year?
from pyspark.sql.functions import max, year
yeardf = df.withColumn ("Year", year (df.Date))
yeardf.groupBy ("Year").max ("High").show ()

# %%What is the average Close for each Calendar Month?
from pyspark.sql.functions import month, avg
dfmonth = df.withColumn ("Month", month (df.Date))
final = dfmonth.groupBy ("Month").avg ("Close").orderBy ("Month")
final.show ()

# %%
