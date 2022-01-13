import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path
import json
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,\
classification_report
import pandas as pd
import itertools
import argparse



def evaluate(model, X_test, y_test):
    yhat = model.predict_classes(X_test)
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



def create_lstm_classifier(x_shape=19,y_shape=1,n_class=1):
    model = Sequential()
    model.add(LSTM(100,input_shape=(x_shape,y_shape)))
    model.add(Dense(n_class, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model


def _lstm(args):
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

    X_train = np.expand_dims(X_train,axis=2)
    X_test = np.expand_dims(X_test,axis=2)

    list_class = list(np.unique(y_test))
    n_class = len(list_class)
    if args.cross_validation:
        clf = KerasClassifier(build_fn=create_lstm_classifier, x_shape=X_train.shape[1],y_shape=X_train.shape[2],n_class=1)
        epochs = [50, 100]
        batches = [16, 32]
        random_grid = dict(epochs=epochs, batch_size=batches)
        rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,
                               n_iter = args.iterations, cv = args.cv, verbose=2, random_state=42, n_jobs = -1)

        rf_random.fit(X_train, y_train)
        best_params = rf_random.best_params_
        model = create_lstm_classifier(X_train.shape[1],X_train.shape[2],1)
        model.fit(X_train,y_train,epochs=best_params['epochs'],batch_size=best_params['batch_size'])

    else:
        model = create_lstm_classifier(X_train.shape[1],X_train.shape[2],1)
        model.fit(X_train,y_train,epochs=100,batch_size=16)

    accuracy,f_score,p_score,r_score,conf_matrix,class_report = evaluate(model, X_test, y_test)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(args.report_output_path,index=True)
    metadata_conf = create_conf_mat_visualization(conf_matrix,list_class,n_class)
    model.save(args.model_output_path)

    with open(args.metadata_output_path, 'w') as f:
        json.dump(metadata_conf, f)

    metadata_metric = create_metric_visualization(accuracy,f_score,p_score,r_score)
    with open(args.metric_output_path, 'w') as f:
        json.dump(metadata_metric, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM classifier')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--cross_validation',type=bool,default=False)
    parser.add_argument('--iterations',type=int,default=2)
    parser.add_argument('--cv',type=int,default=2)
    parser.add_argument('--model_output_path', type=str)
    parser.add_argument('--report_output_path', type=str)
    parser.add_argument('--metadata_output_path',type=str)
    parser.add_argument('--metric_output_path',type=str)

    args = parser.parse_args()

    _lstm(args)







