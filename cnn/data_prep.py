import numpy as np 
import tensorflow as tf


def load_cifar10():

    (images_train, labels_train),(images_test,labels_test) = tf.keras.datasets.cifar10.load_data()

    def normalize(array): 
        return array.astype('float32')/255.0 

    def flatten(threeDarray): 
        return [np.reshape(x,3072) for x in threeDarray] # 32x32x3 for rgb channels

    def one_hot(array,n_classes=10): 
        return np.eye(n_classes)[array].reshape(len(array),n_classes)

    images_train = normalize(images_train)
    images_test = normalize(images_test)
    
    labels_train = one_hot(labels_train)
    labels_test = one_hot(labels_test)