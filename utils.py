# Load libraries
import pickle
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


NUM_FOLDS = 10


def create_models(scale_method=None, balance_method=None):
    """ Cette fonction permet de créer les modèles de base avec les méthodes de normalisation et d'équilibrage de données en paramètres. 
    Par défaut, la fonction crée des modèles sans normalisation et sans équilibrage de données."""
    # The baseline models 
    base_models = [
        ('LR', LogisticRegression(max_iter=1000)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', SVC(probability=True)), # probability=True pour avoir les probabilités de prédiction pour calculer roc_auc_score
        ('XGB', XGBClassifier()),
        ('AB', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('RF', RandomForestClassifier()),
        ('ET', ExtraTreesClassifier())
    ]
    
    # if no scale_method or balance_method is provided, return the baseline models
    if balance_method is None and scale_method is None:
        return base_models
    
    # create the models with the given scale_method and balance_method
    models = []
    for name, model in base_models:
        steps = []
        if scale_method:
            steps.append(('scaler', scale_method()))
        if balance_method:
            steps.append(('balancer', balance_method()))  
        steps.append((name, model))
        models.append(
            (name, imbPipeline(steps=steps)) if balance_method
            else (name, Pipeline(steps=steps))
        )

    return models


def create_std_scaled_models(balance_method=None):
    """ Cette fonction permet de créer les modèles avec la méthode de normalisation StandardScaler et la méthode d'équilibrage de données en paramètres.
    Par défaut, la fonction crée des modèles sans équilibrage de données."""
    return create_models(scale_method=StandardScaler, balance_method=balance_method)


def cross_validation(models, x, y, num_folds=NUM_FOLDS, disable_warnings=True, debug=False):
    """ Cette fonction permet de faire une cross validation sur les modèles en paramètres."""
    if disable_warnings:
        # disable warnings
        warnings.filterwarnings('ignore')
    
    # check if we are in a multiclass classification problem
    multiclass = len(np.unique(y)) > 2
    
    if multiclass:
        scoring = {
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovo', needs_proba=True),
            'accuracy': make_scorer(accuracy_score)
        }
    else:
        scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

    results = []
    for name, model in models:
        if debug:
            print(f"Cross validation for {name}")
        skfold = StratifiedKFold(n_splits=num_folds, shuffle=True)

        cv_results = cross_validate(model, x, y, cv=skfold, scoring=scoring, n_jobs=-1)
        results.append((name,
                        cv_results['test_accuracy'].mean(),
                        cv_results['test_f1'].mean(),
                        cv_results['test_precision'].mean(),
                        cv_results['test_recall'].mean(),
                        cv_results['test_roc_auc'].mean()))
        if debug:
            msg = " Accuracy: %f (%f)\n" % (cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std())
            msg += " F1: %f (%f)\n" % (cv_results['test_f1'].mean(), cv_results['test_f1'].std())
            msg += " Precision: %f (%f)\n" % (cv_results['test_precision'].mean(), cv_results['test_precision'].std())
            msg += " Recall: %f (%f)\n" % (cv_results['test_recall'].mean(), cv_results['test_recall'].std())
            msg += " AUC: %f (%f)" % (cv_results['test_roc_auc'].mean(), cv_results['test_roc_auc'].std())
            print(msg)
            print("-------------------------------------------")
    if disable_warnings:
        # reactivation des warnings
        warnings.filterwarnings('default')
    return results


def compare_algorithms(results, names, title):
    """ Cette fonction permet de comparer les algorithmes en paramètres et de visualiser leurs boxplots.
    results: contient les résultats de la cross validation (accuracy, f1, precision, recall, auc),
    names: contient les noms des algorithmes,
    title: est le titre du graphique."""
    # results contains the results of the cross validation (accuracy, f1, precision, recall, auc)
    # names contains the names of the algorithms
    # title is the title of the graph
    
    # show a boxplot for each metric (accuracy, f1, precision, recall, auc) in a the same graph using subplots
    fig, axes = pyplot.subplots(nrows=1, ncols=5, figsize=(30, 10))
    fig.suptitle(title)
    axes[0].set_title("Accuracy")
    axes[1].set_title("F1")
    axes[2].set_title("Precision")
    axes[3].set_title("Recall")
    axes[4].set_title("AUC")

    # boxplot for accuracy
    axes[0].boxplot([result['test_accuracy'] for result in results])
    axes[0].set_xticklabels(names)
    # boxplot for f1
    axes[1].boxplot([result['test_f1'] for result in results])
    axes[1].set_xticklabels(names)
    # boxplot for precision
    axes[2].boxplot([result['test_precision'] for result in results])
    axes[2].set_xticklabels(names)
    # boxplot for recall
    axes[3].boxplot([result['test_recall'] for result in results])
    axes[3].set_xticklabels(names)
    # boxplot for auc
    axes[4].boxplot([result['test_roc_auc'] for result in results])
    axes[4].set_xticklabels(names)

    pyplot.show()


def grid_search(pipeline, parameters, x, y):
    """ Cette fonction permet de faire une grid search sur les paramètres du grid search en paramètres.
    pipeline: est le pipeline contenant le modèle à tester,
    parameters: sont les paramètres du grid search à tester,
    x: est la matrice des features,
    y: est le vecteur des labels."""
    # if pipeline is an object of type Pipeline, then we need to add the name of the estimator to the parameters keys (e.g. 'LR__C' instead of 'C')
    if isinstance(pipeline, Pipeline):
        parameters = {f'{pipeline.steps[-1][0]}__{key}': value for key, value in parameters.items()}
    grid_obj = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=10, scoring='roc_auc', verbose=1)
    grid_obj.fit(x, y)
    return grid_obj.best_estimator_


def evaluate(model, x_test, y_test):
    """ Cette fonction permet d'évaluer le modèle en paramètres selon les metriques: accuracy, f1, precision, recall et auc.
    model: est le modèle à évaluer,
    x_test: est la matrice des features de test,
    y_test: est le vecteur des labels de test."""
    # check if we are in a multiclass classification problem
    multiclass = len(np.unique(y_test)) > 2

    # calculate acc, f1, precision, recall on the test set
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    average = 'weighted' if multiclass else 'binary'
    f1 = f1_score(y_test, predictions, average=average)
    precision = precision_score(y_test, predictions, average=average)
    recall = recall_score(y_test, predictions, average=average)
    
    # calculate auc
    if len(y_test) > 2:
        probabilities = model.predict_proba(x_test)
        auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
    else:
        probabilities = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    return acc, f1, precision, recall, auc

def summarize_results(results_dict):
    """ Cette fonction permet de résumer les résultats de la cross validation. 
    results_dict: est un dictionnaire contenant les résultats de la cross validation."""
    summary = []

    for dataset_name, results in results_dict.items():
        for model_name, acc, f1, precision, recall, roc_auc in results:
            summary.append((model_name, dataset_name, acc, f1, precision, recall, roc_auc))

    summary = pd.DataFrame(summary, columns=['Model', 'Dataset', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUC'])
    summary.sort_values(by=[ 'Recall','Precision', 'F1','AUC','Accuracy' ], ascending=False, inplace=True, ignore_index=True)
    
    return summary


def combine_X_and_W(X, W, p=1, svd=False):
    """ Cette fonction permet de combiner les matrices X et W en paramètres.
    X: est la matrice des features,
    W: est la matrice de similarité,
    p: est le nombre de puissance de la matrice W,
    svd: si True, on retourne la SVD de la matrice combinée."""
    # Calcul de la matrice diagonale D
    D = np.diag(np.sum(W, axis=1)) # Somme des lignes de W

    # Calcul de l'inverse de D
    D_inv = np.linalg.inv(D)

    # Combinaison des matrices W et X
    # M^p = (D^-1 * W)^p * X
    Mp = np.dot(np.linalg.matrix_power(np.dot(D_inv, W), p), X)
    assert Mp.shape == X.shape

    # Calcul de la SVD de M
    if svd:
        U, _, _ = np.linalg.svd(Mp, full_matrices=False) # U, S, V = np.linalg.svd(Mp)
        assert Mp.shape == U.shape
        return U

    return Mp


def concatenate_X_and_W(X, W):
    """ Cette fonction permet de concaténer les matrices X et W en paramètres.
    X: est la matrice des features,
    W: est la matrice de similarité."""
    # X est une matrice de taille (n, d)
    # W est une matrice de taille (n, n)
    # Concaténation des matrices X et W (colonnes) pour obtenir une matrice de taille (n, d + n)
    M = np.concatenate((X, W), axis=1)
    assert M.shape == (X.shape[0], X.shape[1] + W.shape[1])
    return M
