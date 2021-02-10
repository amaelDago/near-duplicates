import numpy as np
import pandas as pd
import re
import spacy
import json
from unicodedata import normalize
from sklearn.feature_extraction import stop_words
from gensim.models import Doc2Vec
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

#from gensim.models.doc2vec import Doc2Vec, TaggedDocument


nlp = spacy.load("en_core_web_sm")
sw = stop_words.ENGLISH_STOP_WORDS

doc2vec_model = Doc2Vec.load("doc2vecs.pkl")


def levenshtein_relative(word1, word2) : 
    # Calcul la distance entre deux chaine de carcatères
    assert type(word1)==str and type(word2)==str, "Inputs must be string"
    len1 = len(word1)
    len2 = len(word2)
    
    # Zeros Array to fill  
    arr = np.zeros((len1+1, len2+1))
    #cost = 0
    
    arr[:,0] = np.array(list(range(len1+1)))
    arr[0,:] = np.array(list(range(len2+1)))
    
    for i in range(1, len1+1) : 
        for j in range(1, len2+1) : 
            # -1 because python begin count by 0
            if word1[i-1] == word2[j-1] : 
                cost = 0
            else : 
                cost = 1
            
            arr[i,j] = min(arr[i-1,j]+1, arr[i, j-1]+1, arr[i-1, j-1] + cost)
            
            # Word -1 because python count begin by 0
            #if (i>1 and j>1) and word1[i-1]==word2[j-2] and word1[j-2]==word2[j-1] : 
            #    arr[i,j] = min(arr[i,j], arr[i-2,j-2] + cost)
        
    return(arr[len1, len2]/max(len1, len2,1))

def getValuesIfExist(value1, value2) : 
    if isinstance(value1, str) and  isinstance(value2, str): 
        return levenshtein_relative(value1, value2)

    elif isinstance(value1, list) and isinstance(value2, list) :
        return (value1==value2)*1
    else : 
        return 0


# TOkenize for fingerprint
def tokenize(sentence) : 
    tokens = [token.lemma_ for token in nlp(sentence.lower()) if token.lemma_ not in sw]
    return [token for token in tokens if token.isalpha()]

# Tokenize for metadata
def metadataTokenizer(sentence) : 
    assert isinstance(sentence, str), "Input must be a list"

    tokens = [token for token in re.sub(r'[^\w\s]',' ',str(sentence).lower()).split()]
    return tokens


# Choice of fields
fields = ["title","pmId", "volume","typeConditor", "hasFulltext", "xPublicationDate", "documentType", "specialIssue", "pageRange", "xissn","source", "doi", "localRef", "first3authorNames", "sourceUid", "orcId", "duplicateRules", "isNearDuplicate","pii", "idChain", "hasTransDuplicate", "isDeduplicable" ]
title_subfields = ["default", 'monography','meeting','journal']

def getNotice(notice, fields = fields, title_subfields = title_subfields) : 
    
    assert isinstance(notice, (dict, pd.Series)), "Input must be dict"

    dictionary = {}
    dictionary["metadata"] =  [] 
    
    for field in fields : 
        if field == 'title' : 
            dictionary["fingerprint"] = []
            for subfield in title_subfields : 
                print(subfield)
                dictionary["fingerprint"].extend(tokenize(notice["title"][subfield]))
            pass 
        else : 
            try :
                x = notice[field]
                if isinstance(x, list) : 
                    dictionary["metadata"].extend(x)
                else : 
                    dictionary["metadata"].extend([str(x)])
            except : 
                pass
    dictionary['metadata'] = metadataTokenizer(" ".join(dictionary['metadata']))

    return dictionary



def getVector(tokens, model) : 
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    return model.infer_vector(tokens)

def vectorizeNotice(dictionary, doc2vec_model = doc2vec_model) : 
    # Input : a dictionary of notice fitted in getNotice function and differents models
    # otput : a vector representing the notice

    fingerprint = dictionary["fingerprint"]
    metadata = dictionary["metadata"]

    vecFingerprint = getVector(fingerprint, doc2vec_model)
    vecMetadata = getVector(metadata, doc2vec_model)

    return np.concatenate((vecFingerprint,vecMetadata))#, notice[2])

def getVec2Notice(notice, fields = fields, title_subfields = title_subfields , doc2vec_model = doc2vec_model) : 
    vec1 = getNotice(notice, fields, title_subfields)
    return vectorizeNotice(vec1, doc2vec_model)
    #return vec1

def compareNotice(notice1, notice2, fields = fields, title_subfields = title_subfields , doc2vec_model = doc2vec_model) : 
    vec1 = getVec2Notice(notice1, fields, title_subfields, doc2vec_model)
    vec2 = getVec2Notice(notice2, fields, title_subfields, doc2vec_model)
    return  np.abs(vec1- vec2),


def analyseTriplet(triplet, fields = fields, title_subfields = title_subfields , doc2vec_model = doc2vec_model) : 
    vec1 = getVec2Notice(triplet[0], fields, title_subfields, doc2vec_model)
    vec2 = getVec2Notice(triplet[1], fields, title_subfields, doc2vec_model)
    return  np.abs(vec1- vec2), triplet[2]



def comp_(notice1, notice2, fields) : 
    assert isinstance(notice1, (dict, pd.Series)) and isinstance(notice1, (dict, pd.Series)), "Notices must be dicts"

    result = [0] * (len(fields) + 4)
    default = levenshtein_relative(notice1["title"]["default"], notice2["title"]["default"])
    result[0] = default

    monography = levenshtein_relative(notice1["title"]["monography"], notice2["title"]["monography"])
    result[3] = monography

    journal = levenshtein_relative(notice1["title"]["journal"], notice2["title"]["journal"])
    result[4] = journal

    meeting = levenshtein_relative(notice1["title"]["meeting"], notice2["title"]["meeting"])
    result[5] = meeting

    i = 6
    for field in fields : 
        try : 
            result[i]  = getValuesIfExist(notice1[field], notice2[field])
        except : 
            pass
        i+=1

    return result


def get_training_set(data, fields = fields, title_subfields = title_subfields, doc2vec_model = doc2vec_model) : 
    assert isinstance(data, list), "Input must be a list"

    length1 = len(doc2vec_model.docvecs[0])
    
    arr = np.zeros((len(data),(length1 * 2)))
    y = np.zeros(len(data))

    # Loop on entire data
    print("get X and Y for train ...")
    for i,example in tqdm(enumerate(data)) : 
        tuple_ = analyseTriplet(example, fields, title_subfields, doc2vec_model)
        arr[i, :] = tuple_[0]
        y[i] = tuple_[1]

    return arr, y



def score(y, y_pred) : 
    assert len(y)==len(y_pred), "Inputs must be the same length"

    # Confusion matrice
    cf_mat = confusion_matrix(y, y_pred)

    # Get TP, TN, FP, FN
    # We have y = {"Negative" =0, "Postive" = 1}

    TN = cf_mat[0,0]
    TP = cf_mat[1,1]
    FN = cf_mat[0,1]
    FP = cf_mat[1,0]

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FN + FP)

    # Precision
    precision = TP / (TP + FP)

    # Recall
    recall = TP / (TP + FN)

    # F1 Score
    f1 = 2*(accuracy*recall) / (accuracy + recall)

    return {"accuracy": accuracy, "precision":precision, "recall" : recall, "F1_score": f1, "Confusion Matrix" : cf_mat}


def labelling(x, classifier, threshold = 0.75) : 

    # Labelling Data with 3 class (1,0-1) where -1 a value inferior to threshold
    x = x.reshape(1,-1)
    assert len(x)==1, 'Yes'

    prob = classifier.predict_proba(x)

    arg = np.argmax(prob)
    
    if prob[:,1] > 0.75 : 
        return arg
    else : 
        return 0

#classif = np.apply_along_axis(labelling, axis = 1, arr = diff)