import math
import numpy as np
import sys
from sklearn.metrics import roc_curve, auc, precision_recall_curve
np.set_printoptions(suppress=True)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
import xgboost as xgb

def MetricsCalculate(y_true_label, y_predict_label, y_predict_pro):
    metrics_value = []
    confusion = []
    tn, fp, fn, tp = metrics.confusion_matrix(y_true_label, y_predict_label).ravel()
    sn = round(tp / (tp + fn) * 100, 3) if (tp + fn) != 0 else 0
    sp = round(tn / (tn + fp) * 100, 3) if (tn + fp) != 0 else 0
    pre = round(tp / (tp + fp) * 100, 3) if (tp + fp) != 0 else 0
    acc = round((tp + tn) / (tp + fn + tn + fp) * 100, 3)
    mcc = round((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 3) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    f1 = round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) != 0 else 0

    fpr, tpr, thresholds = roc_curve(y_true_label, y_predict_pro)
    precision, recall, thresholds = precision_recall_curve(y_true_label, y_predict_pro)

    auroc = auc(fpr, tpr)
    auprc = auc(recall, precision)

    metrics_value.append(sn)
    metrics_value.append(sp)
    metrics_value.append(pre)
    metrics_value.append(acc)
    metrics_value.append(mcc)
    metrics_value.append(f1)
    metrics_value.append(auroc)
    metrics_value.append(auprc)

    confusion.append(tp)
    confusion.append(fn)
    confusion.append(tn)
    confusion.append(fp)

    return metrics_value, confusion

def TrainBaselineMLModel(train_x, train_y, test_x, test_y, model_obj, cv_fold, **kwargs):

    train_feature_class = []
    train_feature_pro = []
    train_y_new = []
    test_feature_class = {}
    test_feature_pro = {}
    species_feature_class = []
    species_feature_pro = []
    models = []
    arr_valid = []
    folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=100).split(train_x, train_y)

    for i, (train, valid) in enumerate(folds):
        
        # ml_dict = {'NB': GaussianNB(),
        #            'RF': RandomForestClassifier(random_state=100),
        #            'AB': AdaBoostClassifier(random_state=100),
        #            'Bag': BaggingClassifier(random_state=100),
        #            'GBDT': GradientBoostingClassifier(random_state=100),
        #            'DT': DecisionTreeClassifier(random_state=100),
        #            'LR': LogisticRegression(random_state=100, max_iter=10000),
        #            'KNN': KNeighborsClassifier(),
        #            'SVM': svm.SVC(random_state=100, probability=True),
        #            'LDA': LinearDiscriminantAnalysis(),
        #            'LGBM': lgbm.LGBMClassifier(random_state=100),
        #            'XGB': xgb.XGBClassifier(random_state=100, use_label_encoder=False, eval_metric='logloss')}

        
        ml_dict = {'NB': GaussianNB(),
                   'RF': RandomForestClassifier(random_state=100),
                   # 'GBDT': GradientBoostingClassifier(random_state=100),
                   'DT': DecisionTreeClassifier(random_state=100),
                   'LR': LogisticRegression(random_state=100, max_iter=10000),
                   'KNN': KNeighborsClassifier(),
                   'SVM': svm.SVC(random_state=100, probability=True),
                   'LDA': LinearDiscriminantAnalysis(),
                   'LGBM': lgbm.LGBMClassifier(random_state=100),
                   'XGB': xgb.XGBClassifier(random_state=100, use_label_encoder=False, eval_metric='logloss')}

        train_X, train_Y = train_x[train], train_y[train]
        valid_X, valid_Y = train_x[valid], train_y[valid]
        model = ml_dict[model_obj]
        model.fit(train_X, train_Y)
        models.append(model)

        predict_valid_y_class = model.predict(valid_X)
        train_feature_class.extend(predict_valid_y_class)
        predict_valid_y_pro = np.array(model.predict_proba(valid_X))[:, 1]
        train_feature_pro.extend(predict_valid_y_pro)

        train_y_new.extend(valid_Y)
        metrics_value, confusion = MetricsCalculate(valid_Y, predict_valid_y_class, predict_valid_y_pro)
        arr_valid.append(metrics_value)
           
        arr_con = []
        arr_con.append(confusion)

    valid_scores = np.around(np.array(arr_valid).sum(axis=0) / cv_fold, 3)
    valid_scores_std = np.std(np.array(arr_valid), axis=0)
    
    if kwargs['test'] == True:
        print("validation_dataset_scores: ", valid_scores)

    for test_dataset in test_x.keys():
        for ml in models:
            species_feature_class.append(ml.predict(test_x[test_dataset]))
            species_feature_pro.append(np.array(ml.predict_proba(test_x[test_dataset]))[:, 1])
        test_feature_class[test_dataset] = np.around(np.array(species_feature_class).sum(axis=0) / cv_fold, 3)
        test_feature_pro[test_dataset] = np.around(np.array(species_feature_pro).sum(axis=0) / cv_fold, 3)
        species_feature_class.clear()
        species_feature_pro.clear()
        if kwargs['test'] == True:
            predict_y_class = np.where(test_feature_pro[test_dataset] >= 0.5, 1, 0)
            predict_y_pro = test_feature_pro[test_dataset]
            print(test_dataset + ':', MetricsCalculate(test_y[test_dataset], predict_y_class, predict_y_pro))

    return [np.array(train_feature_class), np.array(train_feature_pro), test_feature_class, test_feature_pro, np.array(train_y_new), np.array(valid_scores), np.array(arr_valid), valid_scores_std, models, arr_con]

def GenerateOptimalBaselineMLModel(train_x, train_y, test_x, test_y, model_obj_arr, cv_fold, index, isValidationandTest):
    models = {}

    for model_obj in model_obj_arr:
        results_arr = TrainBaselineMLModel(train_x, train_y, test_x, test_y, model_obj, cv_fold, test=isValidationandTest)
        # models[model_obj] = results_arr
        ##### CSR #######
        # if abs(results_arr[5][0] - results_arr[5][1]) < 3 and results_arr[5][3] > 70:
        print(model_obj)
        print("std",results_arr[7])
        if results_arr[5][3] > 75:
            models[model_obj] = results_arr
            # print("models[model_obj][5][3] = ", models[model_obj][5][3])

    models_ranked = sorted(models.items(), key=lambda d: d[1][5][index], reverse=True)
    
    # for i in range(0, len(models_ranked)):
    #     print("models_ranked[i][0] = ", models_ranked[i][0])
    #     print("models_ranked[i][1][5][3] = ", models_ranked[i][1][5][3])
    
    return models_ranked

def SelfAdaptiveCascadeLayer(train_x, train_y, test_x, test_y, model_obj_arr, cv_fold, index, key, threshold, isValidationandTest):

    models = GenerateOptimalBaselineMLModel(train_x, train_y, test_x, test_y, model_obj_arr, cv_fold, index, isValidationandTest)
    model_name = [(element[0], element[1][5][3]) for element in models]

    train_class_feature_aggregate = [[] for j in range(np.shape(train_y)[0])]
    train_pro_feature_aggregate = [[] for j in range(np.shape(train_y)[0])]
    test_class_feature_aggregate = {}
    test_pro_feature_aggregate = {}
    for test_dataset in test_x.keys():
        test_class_feature_aggregate[test_dataset] = [[] for j in range(np.shape(test_y[test_dataset])[0])]
        test_pro_feature_aggregate[test_dataset] = [[] for j in range(np.shape(test_y[test_dataset])[0])]

    column_name_c = []
    column_name_p = []

    train_y_new = []

    for model in models:

        train_class_feature = model[1][0]
        train_pro_feature = model[1][1]

        train_class_feature_aggregate = np.column_stack((train_class_feature_aggregate, train_class_feature))
        train_pro_feature_aggregate = np.column_stack((train_pro_feature_aggregate, train_pro_feature))

        for test_dataset in test_x.keys():
            test_class_feature = model[1][2][test_dataset]
            test_class_feature_aggregate[test_dataset] = np.column_stack((test_class_feature_aggregate[test_dataset], test_class_feature))

            test_pro_feature = model[1][3][test_dataset]
            test_pro_feature_aggregate[test_dataset] = np.column_stack((test_pro_feature_aggregate[test_dataset], test_pro_feature))

        column_name_c.append('C_' + key + '_' + model[0])
        column_name_p.append('P_' + key + '_' + model[0] + '_1')

        if np.shape(train_y_new)[0] == 0:
            train_y_new = np.array(model[1][4])
        if np.shape(train_y_new)[0] != 0 and (train_y_new == model[1][4]).all():
            train_y_new = np.array(model[1][4])
        else:
            print("The label sets are not same between different models")
            sys.exit()

    if key != '':
        print('Depth:', key, ' Threshold:', threshold)
        print('Selected_Model:', model_name)

    return train_class_feature_aggregate, test_class_feature_aggregate, column_name_c, train_pro_feature_aggregate, test_pro_feature_aggregate, column_name_p, train_y_new, model_name

def CascadeArchitecture(train_x, train_y, test_x, test_y, model_obj_arr, cv_fold, index, threshold, column_name):
    depth_feature = []
    metrics = {}
    dep = 1

    print('train_x: ', np.shape(train_x))

    train_x_original = train_x
    test_x_original = test_x
    column_name_original = column_name

    if dep == 1:
        train_class_feature, test_class_feature, column_name_c, train_pro_feature, test_pro_feature, column_name_p, train_y_new, model_name = SelfAdaptiveCascadeLayer(train_x, train_y, test_x, test_y, model_obj_arr, cv_fold, index, 'dep' + str(dep), threshold, False)
        if len(train_y_new) != 0:

            depth_feature.append([train_x, test_x, train_y, column_name, dep-1])

            train_class_pro_feature = np.column_stack((train_class_feature, train_pro_feature))
            test_class_pro_feature = {}
            for key in test_class_feature.keys():
                test_class_pro_feature[key] = np.column_stack((test_class_feature[key], test_pro_feature[key]))

            index_arr = []
            folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=100).split(train_x_original, train_y)
            for train, valid in folds:
                index_arr.extend(valid)
            train_x_original = train_x_original[index_arr]

            train_x = np.column_stack((train_class_pro_feature, train_x_original))
            train_y = train_y_new

            test_x = {}
            for key in test_class_pro_feature.keys():
                test_x[key] = np.column_stack((test_class_pro_feature[key], test_x_original[key]))

            threshold = sum([element[1] for element in model_name]) / len(model_name)
            column_name = column_name_c + column_name_p + column_name_original

            metrics[dep] = threshold
            print("The mean value of depth " + str(dep) + " is ", metrics[dep])
            print('-------------------------------------------------------------------------------------------')

        else:
            depth_feature.append([train_x, test_x, train_y, column_name, dep - 1])
            metrics[dep] = 0
            print("The threshold of the depth " + str(dep) + " is too large...")
            metrics_ranked = sorted(metrics.items(), key=lambda d: d[1], reverse=True)
            print("The best mean value of depth is ", metrics_ranked[0])
            print('**************************************************************************************************************************')
            return depth_feature, metrics_ranked

    while dep < 500:

        dep += 1

        print('train_x: ', np.shape(train_x))

        train_class_feature, test_class_feature, column_name_c, train_pro_feature, test_pro_feature, column_name_p, train_y_new, model_name = SelfAdaptiveCascadeLayer(train_x, train_y, test_x, test_y, model_obj_arr, cv_fold, index, 'dep' + str(dep), threshold, False)

        if len(train_y_new) != 0:

            threshold = sum([element[1] for element in model_name]) / len(model_name)

            if threshold > metrics[dep - 1]:

                depth_feature.append([train_x, test_x, train_y, column_name, dep-1])

                train_class_pro_feature = np.column_stack((train_class_feature, train_pro_feature))

                test_class_pro_feature = {}
                for key in test_class_feature.keys():
                    test_class_pro_feature[key] = np.column_stack((test_class_feature[key], test_pro_feature[key]))

                index_arr = []
                folds = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=100).split(train_x_original, train_y)
                for train, valid in folds:
                    index_arr.extend(valid)
                train_x_original = train_x_original[index_arr]

                train_x = np.column_stack((train_class_pro_feature, train_x_original))
                train_y = train_y_new

                test_x = {}
                for key in test_class_pro_feature.keys():
                    test_x[key] = np.column_stack((test_class_pro_feature[key], test_x_original[key]))

                column_name = column_name_c + column_name_p + column_name_original
 
                metrics[dep] = threshold

                print("The mean value of depth " + str(dep) + " is ",  metrics[dep])
                print('-------------------------------------------------------------------------------------------')

            else:
                print("The mean value of depth " + str(dep) + " is ",  threshold)
                depth_feature.append([train_x, test_x, train_y, column_name, dep - 1])
                metrics[dep] = 0
                print("The mean value of the depth" + str(dep) + " is less than that of the depth" + str(dep - 1))
                print('-------------------------------------------------------------------------------------------')
                break
        else:
            depth_feature.append([train_x, test_x, train_y, column_name, dep - 1])
            metrics[dep] = 0
            print("The threshold of the depth " + str(dep) + " is too large")
            print('-------------------------------------------------------------------------------------------')
            break

    metrics_ranked = sorted(metrics.items(), key=lambda d: d[1], reverse=True)

    if len(metrics_ranked) != 0:
        print("The best mean value of depth is ", metrics_ranked[0])
        print('**************************************************************************************************************************')

    return depth_feature, metrics_ranked