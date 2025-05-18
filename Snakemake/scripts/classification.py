from snakemake.script import snakemake
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
tqdm.pandas()


spikes = pd.read_pickle(snakemake.input.spikes)
features_simple = pd.read_pickle(snakemake.input.features_simple)
features_ss_spdf = pd.read_pickle(snakemake.input.features_ss_spdf)
features_raw = pd.read_pickle(snakemake.input.features_raw)
dataset_name = snakemake.params.name

# create dict with score funcs for classification 
score_funcs_classification = [
    ("Accuracy", metrics.accuracy_score),
    ("Precision", metrics.precision_score),
    ("Recall", metrics.recall_score),
    ("F1-score", metrics.f1_score)
]

def call_score_func(func, args):
    score = func(*args)
    return round(score, 4)

def call_average_func(func, args):
    score = func(*args, average="macro")
    return round(score, 4)


# create dict to collect the scores for classification
classification_score_dicts = {}
def compute_scores_classification(true_label, predicted_label, feature_name):
    classification_score = {}
    for func in score_funcs_classification:
        if func[0] == "Accuracy":
            classification_score[func[0]] = call_score_func(func[1], [true_label, predicted_label])
        # for other functions each class has its own score, average it 
        else:
            classification_score[func[0]] = call_average_func(func[1], [true_label, predicted_label])
    if not feature_name in classification_score_dicts:
        classification_score_dicts[feature_name] = [classification_score]
    else:
        classification_score_dicts[feature_name].append(classification_score)                                                          
    
# create dict to collect mean of all folds for each classification score
classification_score_mean_dicts = {}
def compute_mean_classificaton():
    for feature in classification_score_dicts:
        mean_dict = {}
        scores = classification_score_dicts[feature][0].keys()
        for score in scores:
            tmp = []
            for i in range(5):
                tmp.append(classification_score_dicts[feature][i][score])
            mean_dict[score] = round(np.mean(tmp),4)
        classification_score_mean_dicts[feature] = mean_dict


# classification method 
def classification(features, key, feature_name, clf="svm"):
    #fig, axs = plt.subplots(3,2, figsize=(10,14))
    # remove NAs and prepare data
    features = features.dropna(axis=1)
    X = features.drop(["track"], axis=1)
    #df1 = features[(features == np.inf).any(axis=1)]
    #features = features.drop(df1.index)
    data_label = features["track"]
    # use 5-fold cross validation and collect the scores in lists
    scores = []
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = data_label.iloc[train_index], data_label.iloc[test_index]
        # train classifier and predict
        if clf == "random":
            clf = RandomForestClassifier() 
        else:
            clf = svm.SVC(kernel='rbf') 
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        # plot results for each fold
        #axs[map_sub[i][0], map_sub[i][1]].set_title(f"Fold: {i}, Accuracy: {round(accuracy_score(np.array(y_test), y_pred), 2)}" )
        #axs[2,1].axis('off')
        #sns.scatterplot(x=features.columns[0], y=features.columns[1], s= 150, hue= np.array(y_test),
         #      palette=track_colors, ax= axs[map_sub[i][0], map_sub[i][1]],
          #   data=X_test).set(title=f"Score {round(accuracy_score(y_test, y_pred),2)}")
        #sns.scatterplot(x=features.columns[0], y=features.columns[1], s= 20, hue= y_pred, ax= axs[map_sub[i][0], map_sub[i][1]],
         #      palette=track_colors,
          #   data=X_test)
        #fig.subplots_adjust(top=0.92)
        # evaluate and plot results
        fig, ax = plt.subplots()
        compute_scores_classification(y_test, y_pred, feature_name)
        cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=clf.classes_)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        for labels in disp.text_.ravel():
            labels.set_fontsize(16)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_xlabel("Predicted label", fontsize=18)
        ax.set_ylabel("True label", fontsize=18)
        plt.close()
    return scores

# classification for PCA features
def classification_PCA(df_features, df_features_notracks, color_palette, dataset_name, key, feature_name, dim= 2):
    pca = PCA(n_components=dim)
    pca_features = pca.fit_transform(df_features_notracks)
    if dim == 2:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2"])
    elif dim == 3:
        pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2" , "PC3"])
    plt.figure()
    fig = sns.scatterplot(x="PC1", y="PC2", s= 100, hue= df_features["track"].to_list(),
                   palette=color_palette,
               data=pca_df).set(title=f"PCA of {key} - {dataset_name}")
    pca_df = pca_df.join( spikes[["track"]])
    pca_df = pca_df[pca_df['track'].notna()]
    scores = classification(pca_df, key, feature_name)
    plt.close()
    
# mapping dict for subplots
map_sub = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)}

# number of clusters
track_names = sorted(spikes['track'].unique())
n_cluster = len(track_names)

# color list to extract colors 
colors = ["tab:blue", "tab:green","tab:orange", "tab:red","tab:cyan", "tab:brown", "tab:pink", "tab:olive"]
# if error, new colors needs to be added
assert len(track_names) < len(colors)
track_colors = {track_names[i]:colors[i] for i in range(len(track_names))}

## classification basic features
features_tracks_basic= features_simple\
    .join(spikes[["track"]])
key = "simple features"
scores = classification(features_tracks_basic, key, "FS1")

    
## classification PCA of complex features
df1 = features_ss_spdf[(features_ss_spdf == np.inf).any(axis=1)]
features_ss_spdf = features_ss_spdf.drop(df1.index)
features_ss_spdf = features_ss_spdf.dropna(axis=1)

features_ss_spdf_fv3 = features_ss_spdf[['F14', 'F18', 'F19']]
features_ss_spdf_fv3_tracks = features_ss_spdf_fv3\
    .join(spikes[["track"]])

features_ss_spdf_tracks = features_ss_spdf\
    .join(spikes[["track"]])


key = "FV3 od ss-spdf features"
scores = classification(features_ss_spdf_fv3_tracks, key, "FS2")

# classification raw complex features
key = "raw ss-spdf"
scores = classification(features_ss_spdf_tracks, key, "FS3")

## classification PCA raw waveform    
features_raw_tracks = features_raw\
    .join(spikes[["track"]])

## classification PCA of raw waveform features
classification_PCA(features_raw_tracks, features_raw, track_colors, dataset_name, key="PCA features of raw signal", feature_name="FS4")
classification_PCA(features_raw_tracks, features_raw, track_colors, dataset_name, key="PCA features of raw signal",feature_name= "FS5", dim=3)


## classification raw waveform
key = "raw waveform features"
scores = classification(features_raw_tracks, key, "FS6")

# compute mean for classification scores (average scores from each fold)
compute_mean_classificaton()

classification_scores = pd.DataFrame(classification_score_mean_dicts)
classification_scores.to_pickle(snakemake.output.classification_results)