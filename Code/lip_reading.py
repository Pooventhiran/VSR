# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 01:03:47 2018

@author: Pooventhiran Naveen
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from models.cnn3d import CNN
from preprocess.makefeatures import FeatureGen

#Dictionary mapping Word IDs to words
labels = {'01': 'begin',
          '02': 'choose',
          '03': 'connection',
          '04': 'navigation',
          '05': 'next',
          '06': 'previous',
          '07': 'start',
          '08': 'stop',
          '09': 'hello',
          '10': 'web'}

#Inputting and extracting features
input_path = os.path.dirname(os.path.join(os.getcwd(), "data"))
feat = FeatureGen(type_ = 'words')
(X, y) = feat.make_features(input_path)
X = X.reshape(X.shape + (1,))
y, label_encoder = feat.make_one_hot_encoding(y)

"""im = np.hstack(train_X[0] * 255)
from PIL import Image
Image.fromarray(im).show()
"""
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.15, random_state=42)

#Training CNN
cnn = CNN()
cnn.build_net()
print(cnn.summary())
cnn.train(train_X, train_y, epochs = 20, validation_split = 0.2)
cnn.plot_graph()

#Testing CNN
pred = cnn.test(test_X)
dec_pred = feat.decode_one_hot(pred, label_encoder).reshape(len(test_X))
dec_truth = feat.decode_one_hot(test_y, label_encoder).reshape(len(test_X))
cnn.evaluate(test_X, test_y) #Evaluating the performance of the result

with open('pred_out.txt' ,'w') as out:
    import sys
    sys_stdout, sys_stderr = sys.stdout, sys.stderr
    sys.stdout = out
    sys.stderr = out
    print('GroundTruth vs Prediction labels')
    for pred_index, ground_index in zip(dec_pred, dec_truth):
        print('%-10s %-5s %-10s'%(labels[ground_index], '=>', labels[pred_index]))

    from collections import defaultdict
    count = defaultdict(int)
    for ground_index in dec_truth:
        count[labels[ground_index]] += 1
    print(count)

    matched = defaultdict(int)
    for pred_index, ground_index in zip(dec_pred, dec_truth):
        if ground_index == pred_index:
            matched[labels[ground_index]] += 1
    print(matched)

    for key in count:
        print('Probability for {}: {}'.format(key, matched[key]/count[key])) if key in matched else 'Probability for {}: 0'.format(key)

sys.stdout, sys.stderr = sys_stdout, sys_stderr
