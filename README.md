Recommendation System on Million Song Dataset

Approach

Users, Songs are collected from train set and test set and are hashed using SHA1 hashing to reduce the size of train and test set together from nearly 3GB to 900 MB

Normalization is done on training and testing set separately with the maximum play count per user

Normalization is done to scale down the ratings  between 0 and 1 i.e., (0,1], which ensures the user’s taste preference is preserved

Alternating Least Square (ALS) model inbuilt in Spark is used for training the recommender model

Parameters used are : Rank : 10, Number of Iterations: 3 , 10, Alpha : 0.01 (for Implicit feedback)
