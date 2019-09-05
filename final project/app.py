from flask import Flask, render_template, request, redirect, Response, jsonify
import numpy as np
import pandas as pd
#import os
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import manifold
import math
from sklearn.preprocessing import StandardScaler

import json


df = pd.read_csv('complete_women_data.csv')

app = Flask(__name__)
#First of all you have to import it from the flask module:
#By default, a route only answers to GET requests. You can use the methods argument of the route() decorator to handle different HTTP methods.
@app.route("/", methods = ['POST', 'GET'])
def index():
    global df
    dataframe = df

    chart_data = dataframe.to_dict(orient='records')
    chart_data = json.dumps(chart_data, indent=2)
    data = {'chart_data': chart_data}
    return render_template("index.html",data=data)


@app.route("/showScree", methods = ['POST', 'GET'])
def showScree():
        #df=pd.read_csv("data_d3.csv")
        global df
        random_points=500
        random_row=np.asarray(random.sample(range(2000),random_points))
        df_random=df.loc[random_row]

		#Normalizing data for Kmeans
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df_normalized = pd.DataFrame(np_scaled)

        if request.args.get('method')=='random':

        ## PCA
		## standardize the data for PCA
		## Random Data-
            standard_data_random = StandardScaler().fit_transform(df_random)

            pca_random = PCA(n_components=10).fit(standard_data_random)
            eigenvalues=pca_random.explained_variance_

		# k means determine k
        #distortions = []
        #K = range(1,20)
        #for k in K:
         #   kmeanModel = KMeans(n_clusters=k).fit(df_normalized)
         #   kmeanModel.fit(df_normalized)
         #   distortions.append(kmeanModel.inertia_)


		# Plot the elbow
        #plt.plot(K, distortions, 'bx-')
        #plt.show()
        else:
            df_stratified=df
            kmeanModel = KMeans(n_clusters=3).fit(df_normalized)
            kmeanModel.fit(df_normalized)
            groups=kmeanModel.labels_
            df_stratified["groups"]=groups
            stratified_points=500
            percentage=stratified_points/len(df)
            pd_result=pd.DataFrame()

            for i in range(0,3):
                x=df_stratified[df_stratified["groups"]==i]
                no_random_points=math.ceil(percentage*len(x))
                random_row=np.asarray(random.sample(range(len(x)),no_random_points))
                points=x.iloc[random_row,:]
                pd_result=pd.concat([pd_result,points])


		## Stratified Data
            standard_data_stratified = StandardScaler().fit_transform(pd_result);

            pca_stratified = PCA(n_components=10).fit(standard_data_stratified)
            eigenvalues=pca_stratified.explained_variance_


        data={'eigenvalues':eigenvalues,'PCA':list(range(1,11))}
        dataframe = pd.DataFrame(data)
        chart_data = dataframe.to_dict(orient='records')
        chart_data = json.dumps(chart_data, indent=2)
        #data = {'chart_data': chart_data}
        #return jsonify(chart_data)
        return chart_data


@app.route("/showPCA", methods = ['POST', 'GET'])
def showPCA():
        global df
        random_points=500
        random_row=np.asarray(random.sample(range(2000),random_points))
        df_random=df.loc[random_row]

		#Normalizing data for Kmeans
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(df)
        df_normalized = pd.DataFrame(np_scaled)

        if request.args.get('method')=='PCAR':

        ## PCA
		## standardize the data for PCA
		## Random Data
            #standard_data_random = StandardScaler().fit_transform(df_random)
            pca_random = PCA(n_components=2).fit_transform(df_random)
            PCA1=pca_random[:,0]
            PCA2=pca_random[:,1]

        else:
            df_stratified=df
            kmeanModel = KMeans(n_clusters=3).fit(df_normalized)
            kmeanModel.fit(df_normalized)
            groups=kmeanModel.labels_
            df_stratified["groups"]=groups
            stratified_points=500
            percentage=stratified_points/len(df)
            pd_result=pd.DataFrame()

            for i in range(0,3):
                x=df_stratified[df_stratified["groups"]==i]
                no_random_points=math.ceil(percentage*len(x))
                random_row=np.asarray(random.sample(range(len(x)),no_random_points))
                points=x.iloc[random_row,:]
                pd_result=pd.concat([pd_result,points])


		## Stratified Data
            pca_stratified = PCA(n_components=2).fit_transform(pd_result)
            PCA1=pca_stratified[:,0]
            PCA2=pca_stratified[:,1]
            groups=pd.Series(pd_result["groups"]).values

        if request.args.get('method')=='PCAR':
            data={'PCA1':PCA1,'PCA2':PCA2}
        else :
            data={'PCA1':PCA1,'PCA2':PCA2,'groups':groups}

        dataframe = pd.DataFrame(data)
        chart_data = dataframe.to_dict(orient='records')
        if request.args.get('method')=='PCAR':
            chart_data = json.dumps(chart_data, indent=2)
        else:
            chart_data = json.dumps(chart_data, indent=3)
        #data = {'chart_data': chart_data}
        #return jsonify(chart_data)
        return chart_data

@app.route("/mds_eucledian", methods = ["GET", "POST"])
def mds_eucledian():
    global df
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    method = request.args.get('method')
    if method == "MDSER":
        random_points=500
        random_row=np.asarray(random.sample(range(2000),random_points))
        df_random=df.loc[random_row]
    else:
        df_stratified=df
        kmeanModel = KMeans(n_clusters=3).fit(df_normalized)
        kmeanModel.fit(df_normalized)
        groups=kmeanModel.labels_
        df_stratified["groups"]=groups
        stratified_points=500
        percentage=stratified_points/len(df)
        pd_result=pd.DataFrame()

        for i in range(0,3):
            x=df_stratified[df_stratified["groups"]==i]
            no_random_points=math.ceil(percentage*len(x))
            random_row=np.asarray(random.sample(range(len(x)),no_random_points))
            points=x.iloc[random_row,:]
            pd_result=pd.concat([pd_result,points])
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    if method=="MDSER":
        similarity = pairwise_distances(df_random, metric='euclidean')
    else:
        temp=pd_result
        similarity = pairwise_distances(temp.drop(['groups'],axis=1), metric='euclidean')
    components = mds_data.fit_transform(similarity)
    if method=="MDSER":
        data={'X':components[:,0],'Y':components[:,1]}
    else:
        data={'X':components[:,0],'Y':components[:,1],'groups':pd_result['groups']}
    dataframe = pd.DataFrame(data)
    chart_data = dataframe.to_dict(orient='records')
    if method=="MDSER":
        chart_data = json.dumps(chart_data, indent=2)
    else:
        chart_data = json.dumps(chart_data, indent=3)
    return chart_data


@app.route("/mds_correlation", methods = ["GET", "POST"])
def mds_correlation():
    global df
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    method = request.args.get('method')
    if method == "MDSCR":
        random_points=500
        random_row=np.asarray(random.sample(range(2000),random_points))
        df_random=df.loc[random_row]
    else:
        df_stratified=df
        kmeanModel = KMeans(n_clusters=3).fit(df_normalized)
        kmeanModel.fit(df_normalized)
        groups=kmeanModel.labels_
        df_stratified["groups"]=groups
        stratified_points=500
        percentage=stratified_points/len(df)
        pd_result=pd.DataFrame()

        for i in range(0,3):
            x=df_stratified[df_stratified["groups"]==i]
            no_random_points=math.ceil(percentage*len(x))
            random_row=np.asarray(random.sample(range(len(x)),no_random_points))
            points=x.iloc[random_row,:]
            pd_result=pd.concat([pd_result,points])
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    if method=="MDSCR":
        similarity = pairwise_distances(df_random, metric='correlation')
    else:
        temp=pd_result
        similarity = pairwise_distances(temp.drop(['groups'],axis=1), metric='correlation')
    components = mds_data.fit_transform(similarity)
    if method=="MDSCR":
        data={'X':components[:,0]*100,'Y':components[:,1]*100}
    else:
        data={'X':components[:,0]*100,'Y':components[:,1]*100,'groups':pd_result['groups']}
    dataframe = pd.DataFrame(data)
    chart_data = dataframe.to_dict(orient='records')
    if method=="MDSCR":
        chart_data = json.dumps(chart_data, indent=2)
    else:
        chart_data = json.dumps(chart_data, indent=3)
    return chart_data


@app.route("/mds_matrix", methods = ["GET", "POST"])
def mds_matrix():
    global df
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    method = request.args.get('method')
    if method == "MDSMR":
        random_points=500
        random_row=np.asarray(random.sample(range(2000),random_points))
        df_random=df.loc[random_row]
    else:
        df_stratified=df
        kmeanModel = KMeans(n_clusters=3).fit(df_normalized)
        kmeanModel.fit(df_normalized)
        groups=kmeanModel.labels_
        df_stratified["groups"]=groups
        stratified_points=500
        percentage=stratified_points/len(df)
        pd_result=pd.DataFrame()

        for i in range(0,3):
            x=df_stratified[df_stratified["groups"]==i]
            no_random_points=math.ceil(percentage*len(x))
            random_row=np.asarray(random.sample(range(len(x)),no_random_points))
            points=x.iloc[random_row,:]
            pd_result=pd.concat([pd_result,points])
            temp= pd_result
    if method == "MDSMR":
        sq_loading = squaredLoadings(df_random, 3)
    else:
        sq_loading = squaredLoadings(temp.drop(['groups'],axis=1), 3)
    top_columns = sorted(range(len(sq_loading)), key=lambda k: sq_loading[k], reverse=True)[:3]
    if method == "MDSMS":
        columns = pd_result[[pd_result.columns[top_columns[0]], pd_result.columns[top_columns[1]], pd_result.columns[top_columns[2]]]]
    else:
        columns = df_random[[df_random.columns[top_columns[0]], df_random.columns[top_columns[1]], df_random.columns[top_columns[2]]]]
    components = pd.DataFrame(columns)
    if method == "MDSMS":
        components = pd.concat([components, pd_result['groups']], axis=1, join='inner')
        components.columns = ["One", "Two", "Three", "Cluster"]
    else:
        components.columns = ["One", "Two", "Three"]
    Data = {"substances": []}
    if method == "MDSMS":
        for index,line in components.iterrows():
            Data["substances"].append({
                    "One": line['One'],
                    "Two": line['Two'],
                    "Three": line['Three'],
                    "Cluster": line['Cluster']
                    })
    else:
        for index,line in components.iterrows():
            Data["substances"].append({
                    "One": line['One'],
                    "Two": line['Two'],
                    "Three": line['Three']
                    })
    result = Data
    return json.dumps(result)

def squaredLoadings(data, k):
    std_data = StandardScaler().fit_transform(data);
    cov_mat = np.cov(std_data.T)
    eValues, eVectors = np.linalg.eig(cov_mat)
    squaredLoadings = []
    for ftrId in range(0, len(eVectors)):
        L = 0
        for compId in range(0, k):
            L = L + eVectors[compId][ftrId] * eVectors[compId][ftrId]
        squaredLoadings.append(L)
    return squaredLoadings

if __name__ == "__main__":
    df = pd.read_csv('data_d3.csv')
    app.run(debug=True)
