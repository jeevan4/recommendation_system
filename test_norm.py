from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext, Row
conf = (SparkConf().set("spark.driver.memory", "5G"))
sc = SparkContext(conf=conf,appName="Challenge-2")
sql = SQLContext(sc)

# Loading Train and Test Data
train_triplets = sc.textFile('file:///users/jeevan4/challenge2/hash_train_triplets.txt').map(lambda x:x.split('\t')).map(lambda x: Row(user=x[0],song=x[1],counts=int(x[2])))
train_visible = sc.textFile('file:///users/jeevan4/challenge2/hash_year1_*visible.txt').map(lambda x:x.split('\t')).map(lambda x: Row(user=x[0],song=x[1],counts=int(x[2])))
test_hidden = sc.textFile('file:///users/jeevan4/challenge2/hash_year1_*hidden.txt').map(lambda x:x.split('\t')).map(lambda x: Row(user=x[0],song=x[1],counts=int(x[2])))
train_data = train_triplets.union(train_visible)

# total_data = train_data.union(test_hidden)

# Creating a dataframe so that they can be used as Structured way 
train_df = sql.createDataFrame(train_data)

test_df = sql.createDataFrame(test_hidden)

# Creating a logical table from dataframe
sql.registerDataFrameAsTable(train_df,'train_data_table')

sql.registerDataFrameAsTable(test_df,'test_data_table')

# Generating Normalized test dataset by calculating max counts from the entire dataset
test_norm_data = sql.sql('Select x.user,song,x.counts/y.maximum from test_data_table x INNER JOIN (select user,max(counts) as maximum from train_data_table group by user) y on x.user = y.user')

# saving the test normalized dataset into filesystem with single partition
test_norm_data.map(lambda x: x.user+','+x.song+','+str(x.c2).coalesce(1).saveAsTextFile('file:///users/jeevan4/challenge2/test_norm')
