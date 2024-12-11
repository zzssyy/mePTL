import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, chi2, f_classif, GenericUnivariateSelect, mutual_info_classif, SelectFwe

def ExtraTreesFeatureRanking(X, Y):

    model = ExtraTreesClassifier(random_state=100)
    model.fit(X, Y)
    importance = np.array(np.abs(model.feature_importances_))
    importance = np.nan_to_num(importance)
    importance = preprocessing.MinMaxScaler().fit_transform(importance.reshape(len(importance), 1))[:, 0]
    indices = np.argsort(importance)[::-1]
    importance_mean = np.mean(importance)
    return indices, importance, importance_mean

def GradientBoostingFeatureRanking(X, Y):

    model = GradientBoostingClassifier(random_state=100)
    model.fit(X, Y)
    importance = np.array(np.abs(model.feature_importances_))
    importance = np.nan_to_num(importance)
    importance = preprocessing.MinMaxScaler().fit_transform(importance.reshape(len(importance), 1))[:, 0]
    indices = np.argsort(importance)[::-1]
    importance_mean = np.mean(importance)
    return indices, importance, importance_mean

def Chi2FeatureRanking(X, Y):

    X = np.array(X)
    # filtered_feature = SelectPercentile(chi2, percentile=100)
    filtered_feature = SelectPercentile(mutual_info_classif, percentile=100)
    filtered_feature.fit(X, Y)
    importance = np.array(np.abs(filtered_feature.scores_))
    importance = np.nan_to_num(importance)
    importance = preprocessing.MinMaxScaler().fit_transform(importance.reshape(len(importance), 1))[:, 0]
    indices = np.argsort(importance)[::-1]
    importance_mean = np.mean(importance)
    return indices, importance, importance_mean

def F_classifFeatureRanking(X, Y):

    X = np.array(X)
    # filtered_feature = SelectPercentile(f_classif, percentile=100)
    filtered_feature = SelectPercentile(chi2, percentile=100)
    filtered_feature.fit(X, Y)
    importance = np.array(np.abs(filtered_feature.scores_))
    importance = np.nan_to_num(importance)
    importance = preprocessing.MinMaxScaler().fit_transform(importance.reshape(len(importance), 1))[:, 0]
    indices = np.argsort(importance)[::-1]
    importance_mean = np.mean(importance)
    return indices, importance, importance_mean

def TPFS_FeatureSelection(X, Y, w, p=1):

    embedding_importance_arr = []
    embedding_rank_arr = []
    statistical_importance_arr = []
    statistical_rank_arr = []

    et_rank = []
    gbdt_rank = []
    chi2_rank = []
    fscore_rank = []

    et_indices, et_importance, et_importance_mean = ExtraTreesFeatureRanking(X, Y)
    gbdt_indices, gbdt_importance, gbdt_importance_mean = GradientBoostingFeatureRanking(X, Y)

    chi2_indices, chi2_importance, chi2_importance_mean = Chi2FeatureRanking(X, Y)
    fscore_indices, fscore_importance, fscore_importance_mean = F_classifFeatureRanking(X, Y)

    m = len(et_indices)

    et_indices = list(map(int, et_indices))
    gbdt_indices = list(map(int, gbdt_indices))

    chi2_indices = list(map(int, chi2_indices))
    fscore_indices = list(map(int, fscore_indices))

    for i in range(m):

        et_rank.append(et_indices.index(i) + 1)
        gbdt_rank.append(gbdt_indices.index(i) + 1)

        chi2_rank.append(chi2_indices.index(i) + 1)
        fscore_rank.append(fscore_indices.index(i) + 1)

    embedding_importance_arr.append(et_importance)
    embedding_importance_arr.append(gbdt_importance)
    embedding_rank_arr.append(et_rank)
    embedding_rank_arr.append(gbdt_rank)

    statistical_importance_arr.append(chi2_importance)
    statistical_importance_arr.append(fscore_importance)
    statistical_rank_arr.append(chi2_rank)
    statistical_rank_arr.append(fscore_rank)

    embedding_importance_average_arr = np.mean(embedding_importance_arr, axis=0)
    embedding_rank_average_arr = np.mean(embedding_rank_arr, axis=0)
    statistical_importance_average_arr = np.mean(statistical_importance_arr, axis=0)
    statistical_rank_average_arr = np.mean(statistical_rank_arr, axis=0)

    lnls_total_average = []
    # for i in range(m):
    #     s = w * ((m - embedding_rank_average_arr[i]) / m * embedding_importance_average_arr[i]) + (1 - w) * ((m - statistical_rank_average_arr[i]) / m * statistical_importance_average_arr[i])
    #     lnls_total_average.append(s)
    
    alpha = 0.5
    for i in range(m):
        s = w * (1 / (1-alpha * (embedding_rank_average_arr[i]/m) ** 0.5) * embedding_importance_average_arr[i]) + (1 - w) * (1 / (1-alpha * (statistical_importance_average_arr[i]/m) ** 0.5) * statistical_importance_average_arr[i])
        lnls_total_average.append(s)
        
    feature_index = np.argsort(lnls_total_average)[::-1]
    num = int(np.round(len(feature_index) * p))
    if num == 0:
        num = 1
    feature_index = feature_index[:num]

    return feature_index