import os
try:
    os.environ["PYSPARK_PYTHON"]="/usr/local/Cellar/python3/3.6.3/bin//python3"
except:
    pass
os.environ["PYSPARK_PYTHON"]="/usr/local/Cellar/python3/3.7.3/bin/python3"
import pyspark
import random
import pandas as pd
import math
import spark_split
import numpy as np
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel


def dcg_at_k(r, k=10, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0

def ndcg_at_k(r, k=10, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0
    return dcg_at_k(r, k, method) / dcg_max

conf = pyspark.SparkConf().setAppName("App")
#############High memory cost IF dataset is large################
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '4G')
        .set('spark.driver.memory', '45G')
        .set('spark.driver.maxResultSize', '10G'))
########################################################
sc = pyspark.SparkContext(conf=conf)

def obsolete():
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

def dat2csv():
    '''
    processing .dat file to csv file
    '''
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('ml-1m/ratings.dat', sep='::',
        header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    ratings.to_csv('ml-1m/ratings.csv', index=False, sep=',')

    ratings_title = ['movieId','title','genres']
    ratings = pd.read_table('ml-1m/movies.dat', sep='::',
        header=None, names=ratings_title, engine='python')
    ratings.to_csv('ml-1m/movies.csv', index=False, sep=',')

large_ratings_file="ml-1m/ratings.csv"
large_ratings_raw_data = sc.textFile(large_ratings_file)
large_ratings_raw_data_header = large_ratings_raw_data.take(1)[0]
large_ratings_data = large_ratings_raw_data.filter(lambda line: line!=large_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
# large_ratings_data = large_ratings_data.map(lambda x: (x[0],(x[1],x[2])))
# ll = sorted(large_ratings_data.groupByKey().mapValues(list).take(10))

# training_RDD, test_RDD = train_large_ratings_data, test_large_ratings_data

# df = large_ratings_data.toDF(["userID", "movieID","rating"])
# df.write.partitionBy("userID")
# llpp = large_ratings_data.partitionBy(283228, lambda k: k[0])

def nonStratifiedSplit():
    '''
    If data set is big enough
    '''
    training_RDD, test_RDD = large_ratings_data.randomSplit([8, 2], seed=0)
    return training_RDD, test_RDD
def stratifiedSplit():
    '''
    split data Each user's data is porpotional in training and testing
    '''
    spark= pyspark.sql.SparkSession.builder.getOrCreate()
    data_DF = spark.createDataFrame(large_ratings_data).toDF('UserID', 'MovieID', 'ratings')
    DFL = spark_split.spark_stratified_split(
        data_DF,
        ratio=0.8,
        min_rating=1,
        filter_by="user",
        col_user='UserID',
        col_item='MovieID',
        col_rating='ratings',
        seed=42,
    )
    training_DF, test_DF = DFL[0], DFL[1]
    if True:
        training_RDD, test_RDD = training_DF.rdd.map(tuple).map(lambda x: (x[0],x[1],x[2])), \
            test_DF.rdd.map(tuple).map(lambda x: (x[0],x[1],x[2]))
    return training_RDD, test_RDD

########################################################
training_RDD, test_RDD = stratifiedSplit()
########################################################

# print('training_RDD\n')
# print(training_RDD.take(10))
# print('test_RDD\n')
# print(test_RDD.take(10))
# print('training_RDD_\n')
# print(training_RDD_.take(10))
# print('test_RDD_\n')
# print(test_RDD_.take(10))

def ALS_fit():
    '''
    Alternating Least Square train the training set and full set,\\
    Save the model.
    '''
    # num_factors=10 num_iter=75 reg=0.05 learn_rate=0.005
    complete_model = ALS.train(training_RDD, 10, seed=5, 
                            iterations=10, lambda_=0.1)
    # # Save and load model
    complete_model.save(sc, "trainsetCollaborativeFilterSmall_")
    complete_model = ALS.train(large_ratings_data, 8, seed=5, 
                            iterations=10, lambda_=0.1)
    # # Save and load model
    complete_model.save(sc, "fullsetCollaborativeFilterSmall_")

fullset_model = MatrixFactorizationModel.load(sc, "fullsetCollaborativeFilterSmall_")

trainset_model = MatrixFactorizationModel.load(sc, "trainsetCollaborativeFilterSmall_")

def eval_rmse_mae():
    '''
    Print RMSE and MAE for testing set
    '''
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    predictions = trainset_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    print('predictions\n')
    print(predictions.take(10))
    print('rates_and_preds\n')
    print(rates_and_preds.take(10))
    rmse_error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    mae_error = rates_and_preds.map(lambda r: abs(r[1][0] - r[1][1])).mean()
    print ('For testing data the RMSE is %s' % (rmse_error))
    print ('For testing data the MAE is %s' % (mae_error))

complete_movies_file="ml-1m/movies.csv"
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]
complete_movies_data = complete_movies_raw_data.filter(lambda line: line!=complete_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),tokens[1],tokens[2])).cache()
complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]),x[1]))

def get_counts_and_averages(ID_and_ratings_tuple):
    '''
    movie reviews number and it's average score
    '''
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings) 

movie_ID_with_ratings_RDD = (large_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

def obsolete2():
    new_user_ID = 1
    # new_user_ratings = large_ratings_data.take(16)
    new_user_ratings = large_ratings_data.filter(lambda x: x[0]==new_user_ID).collect()
    new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)
    new_user_unrated_movies_RDD = (complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids)\
        .map(lambda x: (new_user_ID, x[0])))
    print(new_user_unrated_movies_RDD.take(10))
    new_user_recommendations_RDD = fullset_model.predictAll(new_user_unrated_movies_RDD)
    print(new_user_recommendations_RDD.take(10))

    new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
    print(new_user_recommendations_rating_title_and_count_RDD.take(3))
    recm_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))
    print(recm_RDD.sortByKey(ascending=False).take(10))
    # (149988, (        (6.329273922211785, 'Dossier K. (2009)')  , 3)      )
    # large_ratings_data.union

# For PPT
print('user=12,movies=100 rating: {} \n'.format(trainset_model.predict(12,100)))
def recmForUser(userID=1):
    '''
    Given a userID\\
    Return top 10 movies\\
    Format: (rating,(movieID,movieTitle,movie rated times))
    '''
    # sc.parallelize(trainset_model.recommendProducts(12,10),10).map().join(complete_movies_data)
    new_user_recommendations_rating_RDD = sc.parallelize(trainset_model.recommendProducts(user=userID,num=10))
    new_user_recommendations_rating_RDD = new_user_recommendations_rating_RDD.map(lambda x: (x.product, x.rating))
    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
    recm_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))
    print(recm_RDD.sortByKey(ascending=False).take(10))
    # print(trainset_model.recommendProducts(user=1996,num=10))

Precision = []
Recall = []
Fmeasure = []
ndcg = 0
# For fast test_RDD computing
# test_RDD = test_RDD.map(lambda x: (x[0], (x[1],x[2]))).groupByKey().cache()

def itemRecmMeasure():
    '''
    return measure score\\
    NDCG,Precision,Recall,F2
    '''
    N = 100
    global ndcg
    # For fast test_RDD computing
    test_RDD_ = test_RDD.map(lambda x: (x[0], (x[1],x[2]))).groupByKey().cache()
    for UserID in range(1,N,1) :
        print('process {}\n'.format(UserID))
        dcg = np.zeros(10)
        # For all users recommend, Compute precision recall F2 ...
        try:
            L10 = trainset_model.recommendProducts(user=UserID,num=10)
        except:
            # userID not in model, userID all in test
            continue
        userProductL = [x.product for x in L10]
        # count testing item in it
        # testing items for this user
        # testing_itemsL = test_RDD.filter(lambda x: x[0]==UserID).map(lambda x: x[1]).collect()
        testing_itemsL = list(map(lambda x: x[0],list(test_RDD_.filter(lambda x: x[0]==UserID).map(lambda x: x[1]).collect()[0])))
        # testing_itemsL = list(map(lambda x: x[0],list(test_RDD.lookup(UserID).map(lambda x: x[1]).collect()[0])))
        if len(testing_itemsL)==0:
            # userID all in train
            continue
        # testing_itemsL ratings threshold
        test_hit = set(userProductL).intersection(set(testing_itemsL))
        # for hit_movie in test_hit:
        i=0
        for result in userProductL:
            if result in testing_itemsL:
                rate = trainset_model.predict(UserID,result)
                if rate >= 4:
                    dcg[i] = 2
                elif rate >=3:
                    dcg[i] = 1
                else:
                    dcg[i] = 0
            i += 1
        dcg_user = ndcg_at_k(dcg)
        print('dcg_user {} \n'.format(dcg_user))
        ndcg = ndcg + dcg_user
        
        # percentage testing_itemsL in userProductL
        recmedtestitem4userLen = len(set(userProductL).intersection(set(testing_itemsL)))
        Precision.append(recmedtestitem4userLen / 10)
        # print('Precision {} \n'.format(Precision))
        # calculate all testing items len for this user 
        Recall.append(recmedtestitem4userLen / len(testing_itemsL))
        # compute F
        if Precision[-1] + Recall[-1] == 0:
            Fmeasure.append(-1)
            continue
        Fmeasure.append( (2 * Precision[-1] * Recall[-1]) / (Precision[-1] + Recall[-1]) )
    print('NDCG {} \n'.format(ndcg/N))
    print('Precision {} \n'.format(np.mean(Precision)))
    print('Recall {} \n'.format(np.mean(Recall)))
    print('F2 {} \n'.format(np.mean(list(filter(lambda x: x!=-1,Fmeasure))) ))

# ALS_fit()
eval_rmse_mae()
recmForUser(userID=1996)
itemRecmMeasure()