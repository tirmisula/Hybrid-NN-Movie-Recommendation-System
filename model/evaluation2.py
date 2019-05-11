import pickle
import numpy as np
import pandas as pd
import math
from recommendation import recommend_your_favorite_movie

features,target_values = pickle.load(open('processed_data/test_data.p', 'rb'))
movie_id = features[:,1]
movie_id = np.unique(movie_id)
precision = 0
recall = 0
for user in range(6040):
    results = recommend_your_favorite_movie(user_id=user, top_k=10)
    count = 0
    for result in results:
        movie = result[0]
        if movie in movie_id:
            count=count+1
    precision = precision+count/10
    recall = recall+count/len(movie_id)

precision = precision/6040
recall = recall/6040
print('The value of precision is: ', precision)
print('The value of recall is: ', recall)


