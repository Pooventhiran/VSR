import os
import numpy as np
from sklearn.model_selection import train_test_split
from models.cnn3d import CNN
from preprocess.make_features import FeatureGen

# Dictionary mapping Word IDs to words
labels = {
    "01": "begin",
    "02": "choose",
    "03": "connection",
    "04": "navigation",
    "05": "next",
    "06": "previous",
    "07": "start",
    "08": "stop",
    "09": "hello",
    "10": "web",
}

# Inputting and extracting features
input_path = os.path.join(os.getcwd(), "..", "data")
feat = FeatureGen(type_="words")
(X, y) = feat.make_features(input_path)
X = X.reshape(X.shape + (1,))
y, label_encoder = feat.make_one_hot_encoding(y)

print(f"X and Y shapes: {X.shape} {y.shape}")

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.15, random_state=43
)

# Training CNN
cnn = CNN()
cnn.build_net()
print(cnn.summary())
cnn.train(train_X, train_y, epochs=200, batch_size=16, validation_split=0.2)
cnn.plot_graph()

# Testing CNN
pred = cnn.test(test_X)
cnn.evaluate(test_X, test_y)

# Print confusion matrix and classification report
dec_pred = feat.decode_one_hot(pred, label_encoder).reshape(len(test_X))
dec_truth = feat.decode_one_hot(test_y, label_encoder).reshape(len(test_X))

dec_truth = [labels[x] for x in dec_truth]
dec_pred = [labels[x] for x in dec_pred]
cnn.print_report(dec_truth, dec_pred, labels)
