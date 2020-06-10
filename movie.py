import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.metrics.pairwise import cosine_similarity as cs
cv = cv()
# a = np.array(4000)
import random
def find_title(index):
	try:
		return df[df.index == index]["title"].values[0]
	except:
		pass
def find_ref(title):
	try:
		return df[df.title == title]["index"].values[0]
	except:
		return 1932
df = pd.read_csv("movie.csv")
columns = df.columns
for column in columns:
	df[column] = df[column].fillna('')
	df[column] = df[column].dropna()
def recommend_by_feature(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print("Error:", row)
df["combined_features"] = df.apply(recommend_by_feature,axis=1)
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
count_matrix = cv.fit_transform(df["combined_features"])
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
cosine_sim = cs(count_matrix)
def show(movie): 
	print(movie)
	try:
		a =[]
		i = 0
		movie_to_bot = movie
		ref = find_ref(movie_to_bot)
		top_recommends = sorted(list(enumerate(cosine_sim[ref])),key=lambda x:x[1])
		for element in top_recommends:
			a.append(find_title(element[0]))
			i=i+1
			if i>5:
				break
		return a
	except:
		return ['did not find any recommendations']