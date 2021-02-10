from pymongo import MongoClient
import math
import sys
import streamlit as st 
import json
import numpy as np
from sklearn.svm import SVC
from utils import compareNotice, fields, title_subfields
import pickle


# Model
filename = 'svc_model.pkl'
model = pickle.load(open(filename, 'rb'))




@st.cache(hash_funcs={MongoClient: id})
def get_client():
    return MongoClient("mongodb://localhost:27017")



# Instanciate de collections
client = get_client()
db = client["notice"]

notice = db.notice


st.sidebar.subheader("MongoDB:")
coll_name = st.sidebar.selectbox ("Select collection: ",db.list_collection_names())

#@st.cache
def load_mongo_data(coll_name, idConditor,all_features = False):
    data = db[coll_name].find({"idConditor" : idConditor})[0]
    
    doc = {"title" : data["title"]["default"]}
    xx = [x["idConditor"] for x in data['nearDuplicates']]
    fields = ["first3AuthorNames", "documentType",'pageRange', "source", 'isbn', "typeConditor", "volume", "xissn", "idConditor"]
    for field in fields : 
        try : 
            doc.setdefault(field, "")
            doc[field] = data[field]

        except : 
            pass
    
    for i, x in enumerate(xx) : 
        doc["near Duplicate" + str(i+1)] = x
    
    if all_features : 
        return data
    else : 
        return doc

#@st.cache
def load_mongo_idConditor(coll_name) : 
    data = db[coll_name].find()[:5000]
    #data = [x for x in data]
    return data

# Create a function which load data in cache
data = load_mongo_idConditor("index")
    

option = st.selectbox("Enter a couple of near duplicate", tuple([x for x in data]))
_, id = option.values()

notice1 = load_mongo_data("notice", id[0], all_features = 1)
notice2 = load_mongo_data("notice", id[1], all_features = 1)
d = {1 : "Duplicate", 0: "Near Duplicate"}


#option = st.selectbox("Choisir un couple de doublons", tuple([x for x in list(range(20))]))
button_sent = st.button("SUBMIT")

col1, col2, col3 = st.beta_columns(3)

if button_sent : 
    with col1 : 
        st.write("Showing Data for notice 1")
        try :
            t1 = load_mongo_data("notice", id[0])
            st.json(t1)
            i = 1
        except : 
            st.write("No corresponding values in database")
            i = 0

    with col2 :
        st.write("Showing Data for notice 2")
        try :
            t2 = load_mongo_data("notice", id[1])
            st.json(t2)
            j = 1
        except : 
            st.write("No corresponding value in database")
            j = 0
    
    with col3 : 
        if i + j == 2 :

            vec = np.array(compareNotice(notice1, notice2, fields)).reshape(1,-1)
            #vec = np.random.randn(21).reshape(1,-1)
            pred = np.squeeze(model.predict_proba(vec))
            id = np.argmax(pred)

            if id == 1 : 
                col3.success("Matched !!! ")
                st.write(f"these notice have considered like {d[int(id)]} with a probability of : {round(pred[int(id)],2)}")
            else : 
                col3.error("Not Matched !!!")
                st.write((f"these notice have considered like {d[int(id)]} with a probability of : {round(pred[int(id)],2)}"))

        else : 
            col3.warning('Least one of notices don\'t existing in database')
            

            



