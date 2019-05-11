import os
try:
    os.environ["PYSPARK_PYTHON"]="/usr/local/Cellar/python3/3.6.3/bin//python3"
except:
    pass
os.environ["PYSPARK_PYTHON"]="/usr/local/Cellar/python3/3.7.3/bin/python3"
import pyspark
import random
import math
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel

# sc = pyspark.SparkContext(appName="test")
conf = pyspark.SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '45G')
        .set('spark.driver.maxResultSize', '10G'))
sc = pyspark.SparkContext(conf=conf)

test_large_ratings_file="testset.csv"
test_large_ratings_raw_data = sc.textFile(test_large_ratings_file)
test_large_ratings_raw_data_header = test_large_ratings_raw_data.take(1)[0]
test_large_ratings_data = test_large_ratings_raw_data.filter(lambda line: line!=test_large_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()

train_large_ratings_file="trainset.csv"
train_large_ratings_raw_data = sc.textFile(train_large_ratings_file)
train_large_ratings_raw_data_header = train_large_ratings_raw_data.take(1)[0]
train_large_ratings_data = train_large_ratings_raw_data.filter(lambda line: line!=train_large_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()


large_ratings_file="ml-latest/ratings.csv"
large_ratings_raw_data = sc.textFile(large_ratings_file)
large_ratings_raw_data_header = large_ratings_raw_data.take(1)[0]
large_ratings_data = large_ratings_raw_data.filter(lambda line: line!=large_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
# large_ratings_data = large_ratings_data.map(lambda x: (x[0],(x[1],x[2])))
# ll = sorted(large_ratings_data.groupByKey().mapValues(list).take(10))

training_RDD, test_RDD = train_large_ratings_data, test_large_ratings_data

# df = large_ratings_data.toDF(["userID", "movieID","rating"])
# df.write.partitionBy("userID")
# llpp = large_ratings_data.partitionBy(283228, lambda k: k[0])

training_RDD_, test_RDD_ = large_ratings_data.randomSplit([8, 2], seed=0)

# def randomSplitByUser(df, weights, seed=None):
#     trainingRation = weights[0]
#     fractions = {row['user']: trainingRation for row in df.select('user').distinct().collect()}
#     training = df.sampleBy('user', fractions, seed)
#     testRDD = df.rdd.subtract(training.rdd)
#     test = pyspark.sql.SparkSession.createDataFrame(testRDD, df.schema)
#     return training, test

# training_RDD_, test_RDD_ = randomSplitByUser(large_ratings_data, weights=[0.7, 0.3])

# print('training_RDD\n')
# print(training_RDD.take(10))
# print('test_RDD\n')
# print(test_RDD.take(10))
# print('training_RDD_\n')
# print(training_RDD_.take(10))
# print('test_RDD_\n')
# print(test_RDD_.take(10))

# complete_model = ALS.train(training_RDD, 8, seed=5, 
#                            iterations=10, lambda_=0.1)
# # Save and load model
# complete_model.save(sc, "trainsetCollaborativeFilter")
# complete_model = ALS.train(large_ratings_data, 8, seed=5, 
#                            iterations=10, lambda_=0.1)
# # Save and load model
# complete_model.save(sc, "fullsetCollaborativeFilter")

fullset_model = MatrixFactorizationModel.load(sc, "fullsetCollaborativeFilter")

trainset_model = MatrixFactorizationModel.load(sc, "trainsetCollaborativeFilter")

# test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
# predictions = trainset_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
# rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
# print('predictions\n')
# print(predictions.take(10))
# print('rates_and_preds\n')
# print(rates_and_preds.take(10))
# rmse_error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
# mae_error = rates_and_preds.map(lambda r: abs(r[1][0] - r[1][1])).mean()
# print ('For testing data the RMSE is %s' % (rmse_error))
# print ('For testing data the MAE is %s' % (mae_error))

complete_movies_file="ml-latest/movies.csv"
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings) 
movie_ID_with_ratings_RDD = (large_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


# complete_model = ALS.train(large_ratings_data, 8, seed=5, 
#                            iterations=10, lambda_=0.1)
# # Save and load model
# complete_model.save(sc, "CollaborativeFilter")
# fullset_model = MatrixFactorizationModel.load(sc, "fullsetCollaborativeFilter")

# new_user_ID = 1
# new_user_ratings = large_ratings_data.take(16)
# new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)
# new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids)\
#     .map(lambda x: (new_user_ID, x[0])))
# print(new_user_unrated_movies_RDD.take(10))
# new_user_recommendations_RDD = fullset_model.predictAll(new_user_unrated_movies_RDD)
# print(new_user_recommendations_RDD.take(10))

# new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
# new_user_recommendations_rating_title_and_count_RDD = \
#     new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
# print(new_user_recommendations_rating_title_and_count_RDD.take(3))
# recm_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))
# print(recm_RDD.sortByKey(ascending=False).take(10))
# (149988, (        (6.329273922211785, 'Dossier K. (2009)')  , 3)      )
# large_ratings_data.union


# UserID = range(1,1000,1)
Precision = []
Recall = []
for UserID in range(1,1000,1) :
    # For all users recommend, Compute precision recall F2 ...
    try:
        L10 = trainset_model.recommendProducts(user=UserID,num=10)
    except:
        # userID not in model, userID all in test
        continue
    userProductL = [x.product for x in L10]
    # count testing item in it
    # testing items for this user
    testing_itemsL = test_RDD.filter(lambda x: x[0]==UserID).map(lambda x: x[1]).collect()
    if len(testing_itemsL)==0:
        # userID all in train
        continue
    # percentage testing_itemsL in userProductL
    recmedtestitem4userLen = len(set(userProductL).intersection(testing_itemsL))
    Precision.append(recmedtestitem4userLen / 10)
    # print('Precision {} \n'.format(Precision))
    # calculate all testing items len for this user 
    Recall.append(recmedtestitem4userLen / len(testing_itemsL))

print('Recall {} \n'.format(Recall))