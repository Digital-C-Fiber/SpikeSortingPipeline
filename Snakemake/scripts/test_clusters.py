from snakemake.script import snakemake
import pandas as pd 
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


ap_derivatives = pd.read_pickle(snakemake.input.ap_derivatives)
ap_window_iloc = pd.read_pickle(snakemake.input.ap_window_iloc)
spikes = pd.read_pickle(snakemake.input.spikes)
dataset_name = snakemake.params.name

#snakemake.output.pca_figure 

colors = ["tab:blue", "tab:green","tab:orange", "tab:red","tab:cyan", "tab:brown", "tab:pink", "tab:olive"]
#print(ap_track_window)
track_names = sorted(spikes['track'].unique())
# if error, new colors needs to be added
assert len(track_names) < len(colors)
track_colors = {track_names[i]:colors[i] for i in range(len(track_names))}


spikes_raw = ap_derivatives\
    .join(spikes[["track"]])
df_no_tracks = pd.DataFrame(spikes_raw["raw"].to_list(), columns=np.arange(0,len(spikes_raw["raw"].iloc[0])))
n_cluster = len(track_names)
color_palette = track_colors

cols = list(df_no_tracks.columns)
df1 = df_no_tracks[(df_no_tracks == np.inf).any(axis=1)]
df_no_tracks = df_no_tracks.dropna(axis=1)
# transform to pca features

# create df
for d in [3]: ##3,4,5
    dim = d
    pca = PCA(n_components=dim)
    pca_features = pca.fit_transform(df_no_tracks)
    if dim == 2:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2"])
    elif dim == 3:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2" , "PC3"])
    elif dim == 4:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2" , "PC3", "PC4"])
    elif dim == 5:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2" , "PC3", "PC4", "PC5"])
    le = LabelEncoder()
    le = le.fit(spikes["track"])
    true_label = le.transform(spikes["track"])

    fig = plt.figure(figsize=(8,6)) 
    ax = fig.add_subplot(111, projection='3d')
    #print("plot pca features")
    #sns.scatterplot(x="PC1", y="PC2", s= 100, hue= spikes["track"].to_list(),
    #               palette=color_palette, ax=ax,
    #           data=pca_df).set(title=f"Feature extraction - PCA - {dataset_name}")
    ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], s=10, c=[track_colors[track] for track in spikes["track"].to_list()], marker='o', alpha=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.savefig(snakemake.output.pca_figure, dpi=300)



    
    """
    below_zero_indices = df_no_tracks.index[pca_df["PC1"] < 0].tolist()
    above_zero_indices = df_no_tracks.index[pca_df["PC1"] >= 0].tolist()
    # Highlight points where PC1 is below zero
    plt.figure()
    #print(spikes_raw.loc[above_zero_indices, "raw"].values)
    for row in spikes_raw.loc[above_zero_indices, "raw"]:
        plt.plot(np.arange(30), row, c= "tab:orange", alpha=0.5, label="above zero PC1")
    for row in spikes_raw.loc[below_zero_indices, "raw"]:
        plt.plot(np.arange(30), row, c= "tab:red", alpha=0.5, label="above zero PC1")
    """

    #fig = sns. 
    # clustering
    kmeans_pca = KMeans(n_clusters= n_cluster, init= "k-means++",n_init="auto", random_state=5)
    predicted_label = kmeans_pca.fit_predict(pca_df)
    predicted_label_mapped = le.inverse_transform(predicted_label)
    print(snakemake.params.name, metrics.v_measure_score(true_label, predicted_label))
    # plot first two components
    fig = plt.figure(dpi=300) 
    ax = fig.add_subplot(111, projection='3d')

    #sns.scatterplot(x="PC1", y="PC2", s= 100, ax=ax,
             #    data=pca_df)#.set(title=f"PCA features {dim}-components") 
    
    ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], s=20, alpha=0.4, marker="o", c="tab:grey")
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)

    fig.savefig(snakemake.output.pca_figure_no_color, dpi=300)

    fig = plt.figure(figsize=(8,6)) 
    ax = fig.add_subplot(111, projection='3d')

    #sns.scatterplot(x="PC1", y="PC2", s= 100, hue= list(le.inverse_transform(predicted_label)),
    #               palette=color_palette,ax=ax,
    #             data=pca_df).set(title=f"K-means clustering - PCA features {dim}-components") #, score {accuracy_score(true_label, predicted_label)}") 
    ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], s=10, c=[track_colors[track] for track in list(le.inverse_transform(predicted_label))], marker='o', cmap=color_palette, alpha=0.4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.savefig(snakemake.output.clustering_figure, dpi=300)