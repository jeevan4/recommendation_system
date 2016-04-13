from __future__ import print_function
import sys
import math
from pyspark import SparkContext,SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from operator import add
from pyspark import StorageLevel
import csv

# Apache Spark Config Settings
conf = (SparkConf().setAll([("spark.driver.memory", "5G"),("spark.driver.maxResultSize","2G"),("spark.executor.memory","3G")]))
sc = SparkContext(conf=conf,appName="CollaborativeFiltering")

# Loading normalized train and testdata from hadoop filesystem 
train_norm_data = sc.textFile('file:///users/jeevan4/challenge2/train_norm_data.txt')
test_norm_data = sc.textFile('file:///users/jeevan4/challenge2/test_norm_data.txt')

# Converting the data into Ratings to supply the data into ALS algorithm
train_ratings = train_norm_data.map(lambda x:x.split(",")).map(lambda l: Rating(l[0], l[1], float(l[2]))).persist(StorageLevel.MEMORY_ONLY)
test_ratings = test_norm_data.map(lambda x:x.split(",")).map(lambda l: Rating(l[0], l[1], float(l[2]))).persist(StorageLevel.MEMORY_ONLY)

# ALS algorithm parameters and building recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model = ALS.train(train_ratings, rank, numIterations,0.01)

#Evaluate the model on test data
testdata = test_ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

ratesAndPreds = test_ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)

# Calculating RMSE
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Root Mean Squared Error = " + str(math.sqrt(MSE)))

