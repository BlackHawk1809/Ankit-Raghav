from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity       #Cosine similarity function between two vectors

text = ['London Paris London','Paris Paris London']
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

#print(count_matrix)                     #It will print the cordinate of vectors

#print(count_matrix.toarray())            #it will convert the cordinate of vector to array

similarity_scores = cosine_similarity(count_matrix)         # converting matrix to cosine values

print(similarity_scores)