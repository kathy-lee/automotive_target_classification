from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
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
pca = PCA(n_components=30, svd_solver='randomized', whiten=True).fit(train_data)
train_feature = pca.transform(train_data)
print(train_feature.shape)

print('\nbegin gradient boosting classification.')
param_grid = {'n_estimators': [100, 200, 300],
              'learning_rate': [0.001, 0.01,0.1]}
#classifier = GridSearchCV(GradientBoostingClassifier(), param_grid)
classifier = GridSearchCV(XGBClassifier(), param_grid)
classifier.fit(train_feature, train_label)
print("The best parameters are %s with a score of %0.2f" % (classifier.best_params_, classifier.best_score_))

print('\npredict for test data.')
test_feature = pca.transform(test_data)
print(test_feature.shape)
test_pred = classifier.predict(test_feature)
train_pred = classifier.predict(train_feature)

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


