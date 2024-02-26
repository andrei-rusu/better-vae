import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from skimage.transform import resize
import sys


def read_npy_file(item):
    data = np.load(item.decode())[0]
    data = resize(data, (64, 64), anti_aliasing=True)
    data = np.transpose(data, (2, 0, 1))
    return data.astype(np.float32)


def create_dataset(path, batch_size, limit):
    dataset = tf.data.Dataset.list_files(path, shuffle=True) \
        .take((limit // batch_size) * batch_size) \
        .map(lambda x: tf.py_func(read_npy_file, [x], [tf.float32])) \
        .map(lambda x: x) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return (dataset, iterator, iterator_init_op, get_next)

def get_cifar10_dataset(train, batch_size, limit):
    (x_train, _), (x_test, _) = cifar10.load_data()
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    
    dataset = None 
    if train:
        dataset = tf.data.Dataset.from_tensor_slices(x_train) 
    else:
        dataset = tf.data.Dataset.from_tensor_slices(x_test)
        
    dataset = dataset.take((limit // batch_size) * batch_size) \
            .batch(batch_size) \
            .repeat() \
            .prefetch(2)
    
    print(dataset)
        
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    
    return (dataset, iterator, iterator_init_op, get_next)