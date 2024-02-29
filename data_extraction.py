from pyspark.sql.functions import col

PATH = '/Users/kangjoin/Downloads/used_cars_data.csv'

sc2 = SparkSession.builder.master('local[*]').appName('used_cars_data').getOrCreate()

file_df = sc2.read.csv(PATH,header=True)
file_df.show()

from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType,IntegerType

#df_new = file_df.select(col('price'),col('is_new'),col('mileage') ,col('frame_damaged') ,col('city_fuel_economy') ,col('seller_rating'))

df_new = file_df.select(col('engine_displacement'),col('frame_damaged') ,col('has_accidents') ,col('horsepower') ,col('isCab'),col('is_new'),col('mileage'),col('power'),col('price'),col('seller_rating'))

df_new = df_new.withColumn("engine_displacement",col("engine_displacement").cast(DoubleType()))
df_new = df_new.withColumn("horsepower",col("horsepower").cast(DoubleType()))
df_new = df_new.withColumn("power",col("power").cast(DoubleType()))
df_new = df_new.withColumn("mileage",col("mileage").cast(IntegerType()))
df_new = df_new.withColumn("price",col("price").cast(IntegerType()))
df_new = df_new.withColumn("seller_rating",col("seller_rating").cast(DoubleType()))

df_new.printSchema()
df_new.show()



cols = ['is_new']
col2 = ['frame_damaged','has_accidents','isCab']

df_new= reduce(lambda df_new, c: df_new.withColumn(c, F.when(df_new[c] == 'False', 0).otherwise(1)), cols, df_new)

df_new=  df_new.na.fill(value=0,subset=["mileage"])

df_new = reduce(lambda df_new, c: df_new.withColumn(c, F.when(df_new[c]== 'False', 2).when(df_new[c]== 'True', 0).otherwise(1)), col2, df_new)
df_new= df_new.na.fill(value=0,subset=["engine_displacement"])
df_new= df_new.na.fill(value=0,subset=["horsepower"])
df_new= df_new.na.fill(value=0,subset=["power"])


df_new= df_new.na.fill(value=0,subset=["seller_rating"])
df_new= df_new.na.fill(value=0,subset=["price"])

df_new.show()


df_new = df_new.withColumn("is_new",col("is_new").cast(IntegerType()))
df_new = df_new.withColumn("frame_damaged",col("frame_damaged").cast(IntegerType()))
df_new = df_new.withColumn("has_accidents",col("has_accidents").cast(IntegerType()))
df_new = df_new.withColumn("isCab",col("isCab").cast(IntegerType()))



df_new = df_new.select('*').where(col("price")>0)
df_new = df_new.select('*').where(col("price")<10000000)

df_new = df_new.select('*').where(col("engine_displacement")>0)
df_new = df_new.select('*').where(col("horsepower")>0)

df_new.printSchema()
df_new.show()


#df_new.write.option("header",True).csv(("/Users/kangjoin/Downloads/df_new"))

