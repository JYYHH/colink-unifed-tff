import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, roc_auc_score

def auto_evaluate(keras_model, test_dataset, is_regression = False, compute_auc = True, ignore_pad = False):
    """Evaluate the accuracy of a keras model on a client's test dataset."""
    if is_regression == False :
        if compute_auc == False:
            metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            for client_data in test_dataset:
                for batch in client_data:
                    predictions = keras_model(batch['x'])
                    if ignore_pad:
                        lbl = np.array(batch['y']).reshape(-1)
                        pred = np.array(predictions).reshape(lbl.shape[0], -1)
                        ign = (lbl != 0)
                        metric.update_state(y_true=lbl[ign], y_pred=pred[ign])
                    else:
                        metric.update_state(y_true=batch['y'], y_pred=predictions)
            return metric.result()
        else:
            predict, target = np.array([]).reshape(-1, 2), np.array([]).reshape(-1, 1)
            for client_data in test_dataset:
                for batch in client_data:
                    predictions = keras_model(batch['x'])
                    predict = np.vstack((predict, predictions.numpy().reshape(-1, 2)))
                    target = np.vstack((target, batch['y'].numpy().reshape(-1, 1)))

            return roc_auc_score(target, predict[:,1])
    else :
        metric = tf.keras.metrics.MeanSquaredError()
        for client_data in test_dataset:
            for batch in client_data:
                predictions = keras_model(batch['x'])
                metric.update_state(y_true=batch['y'], y_pred=predictions)
        return metric.result()
