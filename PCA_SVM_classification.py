from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from load_dataset import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 2, 2)

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

print('\nbegin PCA process.')
#pca = PCA(n_components=30, svd_solver='randomized', whiten=True).fit(train_data)
pca = KernelPCA(n_components=30, kernel='rbf').fit(train_data)
train_feature = pca.transform(train_data)
print(train_feature.shape)

print('\nbegin svm.')

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}
svm = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
svm.fit(train_feature, train_label)
print("The best parameters are %s with a score of %0.2f" % (svm.best_params_, svm.best_score_))

#svm = SVC(kernel='rbf', class_weight='balanced', **grid.best_params)
#svm = LinearSVC(C=1,loss='hinge')
#svm = SVC(kernel='poly', gamma='scale')
#svm.fit(train_feature,train_label)

print('\npredict for test data.')
test_feature = pca.transform(test_data)
print(test_feature.shape)
test_pred = svm.predict(test_feature)
train_pred = svm.predict(train_feature)

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


