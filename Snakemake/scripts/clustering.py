from snakemake.script import snakemake
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


spikes = pd.read_pickle(snakemake.input.spikes)
features_simple = pd.read_pickle(snakemake.input.features_simple)
features_ss_spdf = pd.read_pickle(snakemake.input.features_ss_spdf)
features_raw = pd.read_pickle(snakemake.input.features_raw)
dataset_name = snakemake.params.name


# create dict with score funcs for clustering 
score_funcs_clustering = [
    ("V-measure", metrics.v_measure_score),
    ("Rand index", metrics.rand_score),
    ("ARI", metrics.adjusted_rand_score),
    ("MI", metrics.mutual_info_score),
    ("NMI", metrics.normalized_mutual_info_score),
    ("AMI", metrics.adjusted_mutual_info_score),
    ("Homogeneity", metrics.homogeneity_score),
    ("FMI", metrics.fowlkes_mallows_score),
    ("Completeness", metrics.completeness_score),
    ("Accuracy", metrics.accuracy_score)
]

# create dict to collect the scores for clustering
clustering_scores_dict = {}

def call_score_func(func, args):
    score = func(*args)
    return round(score, 4)

def compute_scores_clustering(true_label, predicted_label, feature_name):
    clustering_score = {}
    for func in score_funcs_clustering:
        clustering_score[func[0]] = call_score_func(func[1], [true_label, predicted_label])
    clustering_scores_dict[feature_name] = clustering_score


## clustering PCA features 
def cluster_pca_features(df_tracks, df_no_tracks, dataset_name, color_palette,n_cluster,key, feature_name, dim= 2):
    pca = PCA(n_components=dim)
    # drop na and inf values
    #df_no_tracks = df_no_tracks.dropna(axis=1)
    #df_tracks = df_tracks.dropna(axis=1)
    # transform to pca features
    print(len(df_tracks), len(df_no_tracks))
    pca_features = pca.fit_transform(df_no_tracks)
    # create df
    if dim == 2:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2"])
    elif dim == 3:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2", "PC3"])
    # encode label
    le = LabelEncoder()
    le = le.fit(df_tracks["track"])
    true_label = le.transform(df_tracks["track"])
    # find id outliers, can be dropped by ID
    # using features = features.drop( [{ID}]) 
    # features_basic = features_basic.drop( [{ID}])
    #plt.figure()
    #fig = sns.scatterplot(x="PC1", y="PC2", s= 100, hue= df_tracks["track"].to_list(),
    #               palette=color_palette,
    #           data=pca_df).set(title=f"Feature extraction - PCA - {dataset_name}")
    #for line in range(0,pca_df.shape[0]):
    #    plt.text(pca_df["PC1"][line]+0.2, pca_df["PC2"][line], df_tracks.index[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
    # clustering
    kmeans_pca = KMeans(n_clusters= n_cluster, init= "k-means++",n_init="auto", random_state=random_s)
    predicted_label = kmeans_pca.fit_predict(pca_df)
    predicted_label_mapped = le.inverse_transform(predicted_label)
    #plt.savefig(snakemake.output.cm_clustering_features_basic)
    # plot first two components
    #fig2 = sns.scatterplot(x="PC1", y="PC2", s= 100, hue= list(le.inverse_transform(predicted_label)),
    #               palette=color_palette,
   #              data=pca_df).set(title=f"K-means clustering - PCA features {dim}-components, score {accuracy_score(true_label, predicted_label)}") 
    # evaluate results and plot confusion matrix
    #plt.figure()
    compute_scores_clustering(true_label, predicted_label, feature_name)
    cm = confusion_matrix(df_tracks["track"], predicted_label_mapped, labels=list(set(predicted_label_mapped)))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=list(set(predicted_label_mapped)))
    #disp.plot(cmap=plt.cm.Blues)
    

# set random seed to make results reproducible, changing it changes the result
random_s = 5

# number of clusters
track_names = sorted(spikes['track'].unique())
n_cluster = len(track_names)

# color list to extract colors 
colors = ["tab:blue", "tab:green","tab:orange", "tab:red","tab:cyan", "tab:brown", "tab:pink", "tab:olive"]
# if error, new colors needs to be added
assert len(track_names) < len(colors)
track_colors = {track_names[i]:colors[i] for i in range(len(track_names))}

## clustering basic features
features_basics_tracks = features_simple\
    .join(spikes[["track"]])
cols = list(features_basics_tracks.columns)
features_basics_tracks = features_basics_tracks.dropna(axis=1)

# clustering
#features_basics_tracks_X = features_basics_tracks.drop(["track"], axis=1)
kmeans_pca = KMeans(n_clusters= n_cluster, init= "k-means++", n_init="auto", random_state=random_s)
predicted_label = kmeans_pca.fit_predict(features_simple)

# encode label
le = LabelEncoder()
le = le.fit(features_basics_tracks["track"])
true_label = le.transform(features_basics_tracks["track"])
predicted_label_mapped = le.inverse_transform(predicted_label)


# evaluate results and plot confusion matrix
plt.figure()
compute_scores_clustering(true_label, predicted_label, "FS1")
cm = confusion_matrix(features_basics_tracks["track"], predicted_label_mapped, labels=list(set(predicted_label_mapped)))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=list(set(predicted_label_mapped)))
#disp.plot(cmap=plt.cm.Blues)
#plt.savefig(snakemake.output.cm_clustering_features_basic)


## clustering PCA of SS-SPDF features
df1 = features_ss_spdf[(features_ss_spdf == np.inf).any(axis=1)]
features_ss_spdf = features_ss_spdf.drop(df1.index)
features_ss_spdf = features_ss_spdf.dropna(axis=1)
features_ss_spdf_fv3 = features_ss_spdf[['F14', 'F18', 'F19']]

features_ss_spdf_fv3_tracks = features_ss_spdf_fv3\
    .join(spikes[["track"]])
features_ss_spdf_tracks = features_ss_spdf\
    .join(spikes[["track"]])


## cluster FV3 
kmeans_pca = KMeans(n_clusters= n_cluster, init= "k-means++",n_init="auto", random_state=random_s)
predicted_label = kmeans_pca.fit_predict(features_ss_spdf_fv3)

# encode label
le = LabelEncoder()
le = le.fit(features_ss_spdf_fv3_tracks["track"])
true_label = le.transform(features_ss_spdf_fv3_tracks["track"])
predicted_label_mapped = le.inverse_transform(predicted_label)

# evaluate results and plot confusion matrix
compute_scores_clustering(true_label, predicted_label, "FS2")

## clustering raw SS-SPDF features
kmeans_pca = KMeans(n_clusters= n_cluster, init= "k-means++",n_init="auto", random_state=random_s)
predicted_label = kmeans_pca.fit_predict(features_ss_spdf)

# encode label
le = LabelEncoder()
le = le.fit(features_ss_spdf_tracks["track"])
true_label = le.transform(features_ss_spdf_tracks["track"])
predicted_label_mapped = le.inverse_transform(predicted_label)

# evaluate results and plot confusion matrix
compute_scores_clustering(true_label, predicted_label, "FS3")
cm = confusion_matrix(features_ss_spdf_tracks["track"], predicted_label_mapped, labels=list(set(predicted_label_mapped)))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=list(set(predicted_label_mapped)))
disp.plot(cmap=plt.cm.Blues)

## clustering PCA of raw signal
features_raw_tracks = features_raw\
    .join(spikes[["track"]])

#features_raw_tracks = features_raw_tracks.dropna(axis=0)
#features_raw = features_raw.dropna(axis=0)

#print(features_raw_tracks.head())
#features_raw_X = pd.DataFrame(features_raw["raw"].to_list(), columns=np.arange(0,len(features_raw["raw"].iloc[0])))
cluster_pca_features(features_raw_tracks, features_raw, dataset_name, track_colors, n_cluster, key= "PCA of raw signal", feature_name="FS4")
cluster_pca_features(features_raw_tracks, features_raw, dataset_name, track_colors, n_cluster, key= "PCA of raw signal", feature_name="FS5", dim=3)

## clustering raw signal
kmeans_pca = KMeans(n_clusters= n_cluster,n_init="auto", init= "k-means++", random_state=random_s)
predicted_label = kmeans_pca.fit_predict(features_raw)

# encode label
le = LabelEncoder()
le = le.fit(features_raw_tracks["track"])
true_label = le.transform(features_raw_tracks["track"])
predicted_label_mapped = le.inverse_transform(predicted_label)

# evaluate results and plot confusion matrix
compute_scores_clustering(true_label, predicted_label, "FS6")
plt.figure()
cm = confusion_matrix(features_raw_tracks["track"], predicted_label_mapped, labels=list(set(predicted_label_mapped)))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=list(set(predicted_label_mapped)))
disp.plot(cmap=plt.cm.Blues)


clustering_scores = pd.DataFrame(clustering_scores_dict)
clustering_scores.to_pickle(snakemake.output.clustering_result)
