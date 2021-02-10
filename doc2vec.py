#!/usr/bin/env python

# Import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.feature_extraction import stop_words
import random
from utils import getNotice, compareNotice, fields, title_subfields, get_training_set, score
from tqdm import tqdm
import os
import csv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy


# Instanciate spacy model for english
nlp = spacy.load("en_core_web_sm")

# Get sklearn NLTK stop words
sw = stop_words.ENGLISH_STOP_WORDS


## Data importation
# Load near duplicates file
print('Loading Data ...')
with open("Duplicates.json" , encoding = "utf8", mode = "r") as f : 
    d = json.load(f)

# Index match
with open("DuplicatesIndexMatch.json" , encoding = "utf8", mode = "r") as f : 
    d_index = json.load(f)

# Load near duplicates file
with open("nearDuplicates.json" , encoding = "utf8", mode = "r") as f : 
    nd = json.load(f)

# Index match
with open("nearDuplicatesIndexMatch.json" , encoding = "utf8", mode = "r") as f : 
    nd_flat = json.load(f)
    

print(f"Duplicates shape is {len(nd)} and near duplicates index length is {len(nd_flat)}")


# not duplicates Simulation : We simule tuple(right, left) which represent index of not duplicates data
size = 5000
right = np.random.randint(len(nd), size = size)
left = np.random.randint(len(nd), size = size)

notd = [(x,y) for x, y in zip(right, left)]

# Flat d_index to transform (a, [b,c]) => [(a, b), (a, c)]
d_flat = []
for x, y in d_index.items() : 
    for yy in y :
        d_flat.append((x,yy))

print(f"Flatten Duplicates index have length {len(d_flat)}.")




# Get batch for training 
def get_data(d, d_flat, size, target) : 
    # This function prepares our data for training. 
    # Input : 
    #   d = duplicates files -- dict
    #   d_index : duplicates index --dict
    #   notd : not duoplicates index
    #   length = length of data
    # Output : 
    #   data : a list of tuple(notice1, notice2, target) 
    
    # Shuffle  list

    data = []
    i = 0
    while i<=5000 :
        try : 
            d_ind1, d_ind2 = random.choice(d_flat)
            data.append((d[int(d_ind1)], d[int(d_ind2)]  , target))
        except : 
            pass
        i +=1
    return data

# Get batch
length = 5000
print("Get data for duplicates...")
data_d = get_data(d,d_flat, size = length, target = 1)

print("Get data for not duplicates...")
data_notd = get_data(d,notd, size = length, target = 0)

print("Get data for duplicates...")
data_nd = get_data(nd,nd_flat, size = length, target = 10)

print("FInished !!!")


data = data_d + data_notd
random.shuffle(data)
print(len(data))

# We prepare data for the Doc2vec model
# What do getNotice do ?
# 1 : select and split all features from each notice in 2 groups : fingerprint and metadata
# 2 : tokenize the 2 newer features with two different tokenizer
# 3 : Add each fingerprint and metadata in two list
# 4 : Shuffle list

print("Building data for doc2vecs...")
text = []
for x in data_nd : 

    n1 = getNotice(x[0], fields = fields, title_subfields = title_subfields)
    n2 = getNotice(x[1], fields = fields, title_subfields = title_subfields)

    for y in [n1, n2] : 
        text.append(y["fingerprint"])
        text.append(y["metadata"])

print(len(text))
print("Finished !!!")
random.shuffle(text)


# Using of Doc2vec to get vector from  text using text

vector_sizes = [5*x for x in range(6,15)]
windows = list(range(5,11))
min_counts = list(range(2,8))



if not os.path.isdir("results") : 
    os.mkdir("results")



with open("results/result.txt", mode = "w", encoding = "utf8", newline = "") as f : 
    csvwriter = csv.writer(f)
    csvwriter.writerow(("vector_size",'windows', "min_count", "F1_score"))

    for vector_size in vector_sizes :
        for window in windows : 
            for min_count in min_counts : 

                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(text)]
                doc2vec_model = Doc2Vec(documents, vector_size=vector_size, window=window, min_count=min_count, workers=8)
                logReg = LogisticRegression()
                X,y = get_training_set(data, fields, title_subfields, doc2vec_model)
                X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.5, shuffle= True, stratify = y)
                logReg.fit(X_train, y_train)
                ypred = logReg.predict(X_test)
                scores = score(y_test, ypred)
                print(f"Model with vector size {vector_size}, windows {window} and min_count {min_count} got a f1 score of : {scores['F1_score']}")
                csvwriter.writerow((vector_size, window, min_count, round(scores['F1_score'], 2)))





# Model save
#doc2vecs_model.save("doc2vecs.pkl") 



