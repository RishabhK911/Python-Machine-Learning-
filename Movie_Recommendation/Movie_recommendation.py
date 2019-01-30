# First we will import the necessary libraries 

import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

# Name of the columns listed in the dataset file downloaded 
# Ratings file has 100k ratings data 
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=r_cols)

#Data about the movie
i_cols = ['movie id', 'movie title' ,'release date',
'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('u.item', sep='|', names=i_cols,
encoding='latin-1')

#Build an Interaction Matrix with userid in left and item id in right 
data= ratings.pivot(index = 'user_id', columns ='movie_id', values = 'rating').fillna(0)
data.head()

#create a function to put all the users in dictionary format 
def create_user_dict(interactions):
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict
    
user_dict=create_user_dict(interactions=data)

# create a function to put all itemid with item name in dictionary format 
def create_item_dict(df,id_col,name_col):
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict

movies_dict = create_item_dict(df = items,
                               id_col = 'movie id',
                               name_col = 'movie title')
                               
                               
# The data we have is in dataframe but the Lightgm model accepts only matrix so first convert data values to matrix.                               
x = sps.csr_matrix(data.values)
model = LightFM(no_components=20, loss='warp',learning_rate=0.07)
model.fit(x,epochs=100)


# function that will predict recommendations to user based on user's past or known likings
def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 10, show = True):
    '''
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output - 
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id+1]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index) \
								 .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return return_score_list
    
#predict the users that might like a particular item based on user and item features 
def sample_recommendation_item(model,interactions,item_id,user_dict,item_dict,number_of_user):
    '''
    Funnction to produce a list of top N interested users for a given item
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - item_id = item ID for which we need to generate recommended users
        - user_dict =  Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - number_of_user = Number of users needed as an output
    Expected Output -
        - user_list = List of recommended users 
    '''
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(item_id),n_users)))
    user_list = list(interactions.index[scores.sort_values(ascending=False).head(number_of_user).index])
    return user_list 


rec_list = sample_recommendation_user(model = model, 
                                      interactions = data, 
                                      user_id = 11, 
                                      user_dict = user_dict,
                                      item_dict = movies_dict, 
                                      threshold = 3,
                                      nrec_items = 5,
                                      show = True)

sample_recommendation_item(model = model,
                           interactions = data,
                           item_id = 1,
                           user_dict = user_dict,
                           item_dict = movies_dict,
                           number_of_user = 5)
# Precision is around 82
#Accuracy is around 94  

precision = precision_at_k(model, x,k=10).mean()
accuracy = auc_score(model, x).mean()                               
                               
                               
