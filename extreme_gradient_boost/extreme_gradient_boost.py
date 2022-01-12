import json
import numpy as np
from os.path import join
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,\
classification_report,log_loss,make_scorer
from pathlib import Path
import argparse
import itertools

def evaluate(model, X_test, y_test):
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test,yhat)
    f_score = f1_score(y_test,yhat,average='weighted')
    p_score = precision_score(y_test,yhat,average='weighted')
    r_score = recall_score(y_test,yhat,average='weighted')
    confusion_mat = confusion_matrix(y_test,yhat)
    class_report = classification_report(y_test,yhat,output_dict=True)
    return accuracy,f_score,p_score,r_score,confusion_mat,class_report

def create_conf_mat_visualization(conf_mat,list_class,num_class):
    conf_mat = list(conf_mat.flatten())
    list_class = [str(x) for x in list_class]
    per = itertools.product(list_class, repeat=num_class)
    matrix = []
    cnt = 0
    for p in per:
        tmp = list(p)
        tmp.append(conf_mat[cnt])
        matrix.append(tmp)
        cnt+=1


    df = pd.DataFrame(matrix,columns=['target','predicted','count'])
    print (df)
    metadata = {
    "outputs": [
        {
            "type": "confusion_matrix",
            "format": "csv",
            "schema": [
                {
                    "name": "target",
                    "type": "CATEGORY"
                },
                {
                    "name": "predicted",
                    "type": "CATEGORY"
                },
                {
                    "name": "count",
                    "type": "NUMBER"
                }
            ],
            "source": df.to_csv(header=False, index=False),
            "storage": "inline",
            "labels": list_class
        }
    ]
    }   

    return metadata

def create_metric_visualization(accuracy,f_score,p_score,r_score):
    metrics = {
        'metrics': [
            {
                'name': 'accuracy',
                'numberValue':  accuracy,
                'format': 'PERCENTAGE'
            },
            {
                'name': 'f1-score',
                'numberValue':  f_score,
                'format': 'RAW'       
            },
            {
                'name': 'precision',
                'numberValue': p_score,
                'format': 'RAW',
            },
            {
                'name': 'recall',
                'numberValue': r_score,
                'format': 'RAW'
            }
        ]
    }
    return metrics

    


def _xgboost(args):
    Path(args.model_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metric_output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_path) as f:
        data = json.load(f)

    data = json.loads(data)

    X_train = np.array(data['x_train'])
    y_train = np.array(data['y_train'])
    
    X_test = np.array(data['x_test'])
    y_test = np.array(data['y_test'])

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    eta = [0.01,0.05,0.1]
    subsample = [0.1,0.2,0.5]
    colsample_bytree = [0.1,0.2,0.5]
    random_grid = {'n_estimators': n_estimators,
                'eta': eta,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                }


    clf = XGBClassifier()
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,
                               n_iter = 1, cv = 2, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_train, y_train,eval_metric='logloss')
    best_grid = rf_random.best_estimator_
    best_grid.fit(X_train,y_train,eval_metric='logloss')
    accuracy,f_score,p_score,r_score,conf_matrix,class_report = evaluate(best_grid, X_test, y_test)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(args.report_output_path,index=True)
    list_class = np.unique(y_test)
    num_class = len(list_class)
    metadata_conf = create_conf_mat_visualization(conf_matrix,list_class,num_class)
    dump(clf, args.model_output_path)

    with open(args.metadata_output_path, 'w') as f:
        json.dump(metadata_conf, f)

    metadata_metric = create_metric_visualization(accuracy,f_score,p_score,r_score)
    with open(args.metric_output_path, 'w') as f:
        json.dump(metadata_metric, f)


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--model_output_path', type=str)
    parser.add_argument('--report_output_path', type=str)
    parser.add_argument('--metadata_output_path',type=str,default='/mlpipeline-ui-metadata.json')
    parser.add_argument('--metric_output_path',type=str)

    args = parser.parse_args()

    _xgboost(args)
