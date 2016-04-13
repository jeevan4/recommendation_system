**Recommendation System on Million Song Dataset**

Implemented Collaborative Filter Recommendation system on 1M song dataset that contains user-song play counts. Used Alternating Least Square (ALS) model inbuilt in Apache Spark’s MLlib for training the recommendation model after doing appropriate normalizations. Achieved RMSE of 0.082.

*Training Set:* http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip

*Testing Set:* http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/EvalDataYear1MSDWebsite.zip


**Approach**

1. Users, Songs are collected from train set and test set and are hashed using SHA1 hashing to reduce the size of train and test set together from nearly 3GB to 900 MB

2. Normalization is done on training and testing set separately with the maximum play count per user

3. Normalization is done to scale down the ratings  between 0 and 1 i.e., (0,1], which ensures the user’s taste preference is preserved

4. Alternating Least Square (ALS) model inbuilt in Spark is used for training the recommender model

5. Parameters used are : Rank : 10, Number of Iterations: 3 , 10, Alpha : 0.01 (for Implicit feedback)

