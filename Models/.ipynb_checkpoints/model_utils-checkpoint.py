import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import pairwise_distances
from sklearn.cluster import *

def plt_elbow_for_clustering(df, col_names, method = 'kmeans', max_grps = 12):
    distortions = []
    r = range(1,max_grps + 1)
    for k in r:
        if method == 'kmeans':
            model = KMeans(n_clusters = k)
        model.fit(df[[c for c in col_names]])
        distortions.append(model.inertia_)
    plt.figure(figsize=(8,6))
    plt.plot(r, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('WSS Error')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
def plt_geog_cluster(df, col_names, labels, map_img):
    df = df[[c for c in col_names]]
    df['cluster'] = labels
    ax = df.plot(
        kind="scatter", 
        x="longitude", 
        y="latitude", 
        figsize=(18,12),
        c="cluster", 
        cmap=plt.get_cmap("bwr"),
        colorbar=True, 
        alpha=0.99,
    )
    # use our map with it's bounding coordinates
    plt.imshow(map_img, extent=[103.5,104,1.15, 1.50], alpha=0.6)            
    # add axis labels
    plt.ylabel("Latitude", fontsize=20)
    plt.xlabel("Longitude", fontsize=20)
    # set the min/max axis values - these must be the same as above
    plt.ylim(1.15, 1.50)
    plt.xlim(103.5, 104)
    plt.legend(fontsize=20)
    plt.show()
    
def cluster_ts(df, n_clusters, dist_metric='correlation'):
    tsclust = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity=dist_metric)
    tsclust.fit(df)
    labels = tsclust.labels_
    return labels

def cluster_geog(df, col_names, n_clusters):
    df = df[[c for c in col_names]]
    model = KMeans(n_clusters = n_clusters, init ='k-means++')
    labels = model.fit_predict(df)
    return labels
        
def get_cluster_loc(cluster_labels):
    num_clusters = len(set(cluster_labels))
    cluster_loc = dict()
    for i in range(num_clusters):
        cluster_loc[i] = list()
    for ind, val in enumerate(cluster_labels):
        cluster_loc[val].append(ind)
    return cluster_loc

def form_cluster_grp(df, cluster_labels, method='sum'):
    assert len(df) == len(cluster_labels)
    n_clusters = len(set(cluster_labels))
    grps_name = ["Cluster_" + str(i) for i in cluster_labels]
    df['cluster'] = grps_name
    if method == 'sum':  # sum the demand of every location in the cluster
        cluster_df = df.groupby("cluster").sum()
    return cluster_df

def get_past_distr(train_df, cluster_loc):
    train_df_sum = train_df.sum(axis=1)
    distr_dict = dict()
    num_clusters = len(cluster_loc.keys())
    for clust in range(num_clusters):
        distr_dict[clust] = list()
        for locs in cluster_loc[clust]:
            distr_dict[clust].append(train_df_sum[locs])
        if sum(distr_dict[clust]) > 0:
            distr_dict[clust] = np.divide(distr_dict[clust], sum(distr_dict[clust]))
    return distr_dict

def assign_demand_to_loc(pred_clust_demand, distr_dict, cluster_loc):
    '''
    pred_demand is df where:
    rows: cluster no.
    cols: predicted at timestep i
    '''
    pred_loc_demand = dict()
    num_periods = pred_clust_demand.shape[1]  #no. of columns
    for period in range(num_periods):
        for clust in distr_dict.keys():
            clust_demand = pred_clust_demand.iloc[clust, period]
            distr = distr_dict[clust]
            loc_demand = distr * clust_demand
            for ind, dem in enumerate(loc_demand):
                loc = cluster_loc[clust][ind]
                if loc not in pred_loc_demand.keys():
                    pred_loc_demand[loc] = list()
                pred_loc_demand[loc].append(dem)
    pred_loc_demand = OrderedDict(sorted(pred_loc_demand.items(), key=lambda t: t[0]))
    return pd.DataFrame.from_dict(pred_loc_demand, orient='index')