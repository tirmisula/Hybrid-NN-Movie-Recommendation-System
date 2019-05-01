import pickle
import pandas as pd
import math
from recommendation import rating_movie

features,target_values = pickle.load(open('processed_data/test_data.p', 'rb'))
maeSum=0
rmseSum=0
training=50000
for i in range(training):
    user_id = features[:,0][i]
    movie_id = features[:,1][i]
    prediction_rating = rating_movie(user_id, movie_id)[0][0]
    test_rating = target_values[i]
    maeSum = maeSum+abs(test_rating-prediction_rating)
    rmseSum = rmseSum+math.pow(test_rating-prediction_rating,2)

print('The value of MAE is:', maeSum/training)
print('The value of RMSE is:', math.sqrt(rmseSum/training))


