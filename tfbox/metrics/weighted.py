import tensorflow as tf
import tfbox.utils.helpers as h


def get(metric):
    """ get keras metric by name
    Args:

        - metric<str|metric>:
            * snake-case strings will be turned to camel-case
            * if metric is not a string the passed metric will be returned
              with the assumption that it is already a keras metric class

    Return: metric class 
    """
    if isinstance(metric,str):
        metric=h.camel(metric)
        metric=getattr(tf.keras.metrics,metric)
    return metric


def weighted(weights,metric='categorical_accuracy'):
    """ weighted metric
    Args:

        - weights<list>:
            * a weight for each value in y_true
        - metric<str|metric>:
            * snake-case strings will be turned to camel-case
            * if metric is not a string the passed metric will be returned
              with the assumption that it is already a keras metric class

    Return: (weighted) metric instance 
    """
    metric=get(metric)()
    def _weighted_metric(y_true,y_pred):
        sample_weight=tf.reduce_sum(weights*y_true, axis=-1)
        metric.reset_states()
        metric.update_state(y_true,y_pred,sample_weight=sample_weight)
        return metric.result()
    _weighted_metric.name=f'{metric.name}-w'
    return _weighted_metric
 

def subset(ignore_labels,labels,metric='categorical_accuracy'):
    """ metric which ignores labels (wrapper for `weighted`)
    Args:

        - ignore_labels<list|int|float>:
            * label(s) to not include in metric calculation
        - labels<list|int|float>:
            * if (int|float) assumes labels are range(labels)
        - metric<str|metric>:
            * snake-case strings will be turned to camel-case
            * if metric is not a string the passed metric will be returned
              with the assumption that it is already a keras metric class

    Return: (weighted) metric instance 
    """
    if isinstance(ignore_labels,(int,float)):
        ignore_labels=[int(ignore_labels)]
    if isinstance(labels,(int,float)):
        labels=list(range(labels))
    weights=[int(i not in ignore_labels) for i in labels]
    return weighted(weights,metric=metric)
 


