import json
import logging
import os
from base64 import b64decode
from io import BytesIO
from PIL import Image

import numpy as np

def read_data_femnist(train_data_dir, test_data_dir):
    train_data,test_data = [],[]
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') and f[0]!='_']
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        my_data = np.array(json.load(open(file_path, 'r'))["records"])
        train_data.append(my_data)
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') and f[0]!='_']
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        my_data = np.array(json.load(open(file_path, 'r'))["records"])
        test_data.append(my_data)


    return train_data, test_data

def read_data_celeba(train_data_dir, test_data_dir):
    train_data,test_data = [],[]
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') and f[0]!='_']
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        data = json.load(open(file_path, 'r'))["records"]
        my_data = []
        for x in data:
            y = np.array(x[0]).reshape(1)
            PIL_image = Image.open(BytesIO(b64decode(x[1]))).crop((0, 20, 178, 198)).resize((224, 224))
            X = np.array(PIL_image).reshape(-1) / 255.0
            my_data.append(np.concatenate((y, X), axis = 0))
        
        train_data.append(np.array(my_data))


    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') and f[0]!='_']
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        data = json.load(open(file_path, 'r'))["records"]
        my_data = []
        for x in data:
            y = np.array(x[0]).reshape(1)
            PIL_image = Image.open(BytesIO(b64decode(x[1]))).crop((0, 20, 178, 198)).resize((224, 224))
            X = np.array(PIL_image).reshape(-1) / 255.0
            my_data.append(np.concatenate((y, X), axis = 0))
        
        test_data.append(np.array(my_data))


    return train_data, test_data  

VOCAB_SIZE = 0
my_vocab = {}

def word_index_train(word):
    global VOCAB_SIZE
    global my_vocab

    if word in my_vocab:
        return my_vocab[word]
    else:
        my_vocab[word] = VOCAB_SIZE
        VOCAB_SIZE += 1
        return VOCAB_SIZE - 1

def word_index_test(word):
    global VOCAB_SIZE
    global my_vocab

    if word in my_vocab:
        return my_vocab[word]
    else:
        return 3

def trans_train(x):
    return np.array([[word_index_train(word) for word in x_item] for x_item in x], dtype = np.int64)

def trans_test(x):
    return np.array([[word_index_test(word) for word in x_item] for x_item in x], dtype = np.int64)


def read_data_reddit(train_data_dir, test_data_dir):
    global VOCAB_SIZE
    VOCAB_SIZE = 4
    global my_vocab
    my_vocab = {'<PAD>' : 0, '<BOS>' : 1, '<EOS>' : 2, '<OOV>' : 3}


    train_data, test_data = [], []
    
    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') and f[0]!='_']
    for f in train_files:
        nx, ny = np.array([], dtype = np.int64).reshape(-1, 10), np.array([], dtype = np.int64).reshape(-1, 10)

        file_path = os.path.join(train_data_dir, f)
        my_data = json.load(open(file_path, 'r'))["records"]

        for x in my_data:
            ny = np.vstack((ny, trans_train(x[0]['target_tokens'])))
            nx = np.vstack((nx, trans_train(x[1])))

        
        train_data.append(np.stack((nx, ny), axis = 0).transpose(1, 0, 2))
    
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') and f[0]!='_']
    for f in test_files:
        nx, ny = np.array([], dtype = np.int64).reshape(-1, 10), np.array([], dtype = np.int64).reshape(-1, 10)

        file_path = os.path.join(test_data_dir, f)
        my_data = json.load(open(file_path, 'r'))["records"]

        for x in my_data:
            ny = np.vstack((ny, trans_test(x[0]['target_tokens'])))
            nx = np.vstack((nx, trans_test(x[1])))

        test_data.append(np.stack((nx, ny), axis = 0).transpose(1, 0, 2))




    return train_data, test_data, VOCAB_SIZE