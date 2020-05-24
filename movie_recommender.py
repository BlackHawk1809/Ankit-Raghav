import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

##### helper functions  use them when needed  #####
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

####################################################

## Step 1 : Read the Csv file
df = pd.read_csv("C://Users//hp-p//Desktop//movie_dataset.csv")
print (df.columns)

## Step 2:   Select the features
features = ['keywords','cast','genres','director']

## step 3: Create a column in df which combines all selected features
for feature in features:
	df[feature] = df[feature].fillna('')                 # it will fill all the NAN values

def combine_features(row):
	try:
		return row['keywords'] + " " +row['cast']+ " " +row['genres']+ " " +row['director']
	except:
		print("Error : ",row)

df["combined_features"] = df.apply(combine_features,axis = 1)

print("combined_features :", df["combined_features"].head()) 


## Step 4: Create Count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

## Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix) 
movie_user_likes = "Avatar"


## Step 6 : Get Index of the movie from the title
movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))

#Step 7: Get a lost of a similar movies in decendening orfer of similarity score
sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1],reverse=True)

## Step 8: print titles of first 50 movies
i=0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i+=1
	if i>50:
		break