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

# Creating a Dataframe object to query on SQL context
train_df = sql.createDataFrame(train_data)

test_df = sql.createDataFrame(test_hidden)

# registering a logical table on dataframe to perfrom sql like queries
sql.registerDataFrameAsTable(train_df,'train_data_table')

sql.registerDataFrameAsTable(test_df,'test_data_table')

# calculating the normalized data
train_norm_data = sql.sql('Select x.user,song,x.counts/y.maximum from train_data_table x INNER JOIN (select user,max(counts) as maximum from train_data_table group by user) y on x.user = y.user')
# saving the normalized train data into filesystem with a single partition, so that the data can be saved in one large file
train_norm_data.map(lambda x: x.user+','+x.song+','+str(x.c2)).coalesce(1).saveAsTextFile('file:///users/jeevan4/challenge2/train_norm_data')
