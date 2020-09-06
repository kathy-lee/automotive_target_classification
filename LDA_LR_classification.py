from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from train_helper import load_data


data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 2, 2)

print('\nbegin LDA process.')
lda = LinearDiscriminantAnalysis(n_components=4, solver='svd').fit(train_data, train_label)
train_feature = lda.transform(train_data)
print(train_feature.shape)

print('\nbegin Logistic Regression.')
lr = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=100)
lr.fit(train_feature,train_label)

print('\npredict for test data.')
test_feature = lda.transform(test_data)
print(test_feature.shape)
test_pred = lr.predict(test_feature)
train_pred = lr.predict(train_feature)

print('\nevaluate the prediction(train data).')
conf = confusion_matrix(train_label, train_pred)
print(conf)
score_precision = precision_score(train_label, train_pred, average=None)
score_recall = recall_score(train_label, train_pred, average=None)
print(score_precision)
print(score_recall)

print('\nevaluate the prediction(test data).')
conf = confusion_matrix(test_label, test_pred)
print(conf)
score_precision = precision_score(test_label, test_pred, average=None)
score_recall = recall_score(test_label, test_pred, average=None)
print(score_precision)
print(score_recall)

plt.plot(lda.explained_variance_ratio_)
plt.plot(np.cumsum(lda.explained_variance_ratio_))
plt.show()
