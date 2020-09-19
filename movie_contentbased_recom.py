# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# %%
df = pd.read_csv('movie_dataset_CH.csv')


# %%
df.head()


# %%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# %%
text = ["London Paris London","Paris Paris London"]
counts = vectorizer.fit_transform(text)


# %%
counts.toarray()


# %%
cosine_similarity(counts)


# %%
df.shape


# %%
df.info()


# %%
features = ["keywords","cast","genres","director"]

for feature in features:
    df[feature] = df[feature].fillna('')


# %%
def combine_features(row):

    return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]


# %%
combine_features(df.iloc[0,:])


# %%
df["combined_features"] = df.apply(combine_features,axis=1)


# %%
df.iloc[:5,-1]


# %%
text = []


# %%
for key in df.loc[:,'combined_features']:
    text.append(key)


# %%
len(text)


# %%
counts = vectorizer.fit_transform(df['combined_features'])


# %%
cosine_model = cosine_similarity(counts)


# %%
cosine_similarity(counts)


# %%
cosine_model_df = pd.DataFrame(cosine_model,index=movies_list,columns=movies_list)
cosine_model_df.head()


# %%
def make_recommendations(movie_user_likes):
    return cosine_model_df[movie_user_likes.lower()].sort_values(ascending=False)[1:20]


# %%
print('\n'.join(list(make_recommendations('avatar').index)))


# %%
movies_list = [s.lower() for s in df['title']]


# %%
s = 'SAI'
s.lower()


# %%
movies_list[:5]


# %%
def find_movies(movie):
    list_mov = []
    for k in movies_list:
        if k.find(movie) != -1:
            list_mov.append(k)
    return list_mov   


# %%
if len(find_movies('wall')) > 0:
    print(find_movies('wall'))
else:
    print('Check the spelling or Enter movies released before 2017')


# %%
import joblib


# %%
joblib.dump(cosine_model,'cosine_model')


# %%
joblib.dump(cosine_model_df,'movie_recom_df')


# %%
import joblib
import pandas as pd


# %%
cosine_model = joblib.load('cosine_model')


# %%
movie_recom_df = joblib.load('movie_recom_df')


# %%
joblib.dump(movies_list,'movies_list')


# %%
cosine_model_df = joblib.load('movie_recom_df')


# %%
def samplefunc(col):
    try:
        df[col]
    except:
        print('Not found')


# %%
samplefunc('dasd')


# %%
joblib.dump(find_movies,'find_movies')


# %%
joblib.dump(make_recommendations,'make_recommendations')





