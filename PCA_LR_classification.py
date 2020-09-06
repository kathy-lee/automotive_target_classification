from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from train_helper import load_data

data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 2, 2)

print('\nbegin PCA process.')
pca = PCA(n_components=1000, svd_solver='randomized', whiten=True).fit(train_data)
train_feature = pca.transform(train_data)
print(train_feature.shape)

print('\nbegin Logistic Regression.')
lr = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=200)
lr.fit(train_feature,train_label)

print('\npredict for test data.')
test_feature = pca.transform(test_data)
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

plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

# param_grid = {'C': [1e3], 'gamma': [ 0.0001]}
# svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
# svm = svm.fit(train_data, train_label)

