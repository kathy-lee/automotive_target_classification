from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from train_helper import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



data_dir = '/home/kangle/dataset/PedBicCarData'
train_data, train_label, test_data, test_label = load_data(data_dir, 4, 5)

print("Data sample distribution in training set: %d %d %d %d %d\n" % (np.count_nonzero(train_label==1),
      np.count_nonzero(train_label==2), np.count_nonzero(train_label==3),
      np.count_nonzero(train_label==4), np.count_nonzero(train_label==5)))
print("Data sample distribution in test set: %d %d %d %d %d\n" % (np.count_nonzero(test_label==1),
      np.count_nonzero(test_label==2), np.count_nonzero(test_label==3),
      np.count_nonzero(test_label==4), np.count_nonzero(test_label==5)))

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# print('\nbegin LLE process.')
# lle = LocallyLinearEmbedding(n_components=20,n_neighbors=10,n_jobs=-1)
# train_feature = lle.fit_transform(train_data)
# print(train_feature.shape)
#
# print(lle.reconstruction_error_)

train_feature = train_data
test_feature = test_data

print('\nbegin gradient boosting classification.')
# # param_grid = {'n_estimators': [300,400],
# #               'learning_rate': [0.1],
# #               'max_depth': [5]}
# #classifier = GridSearchCV(GradientBoostingClassifier(), param_grid)
# #classifier = GridSearchCV(XGBClassifier(), param_grid)
# classifier = XGBClassifier( learning_rate =0.1,
#  n_estimators=300,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# classifier.fit(train_feature, train_label)
# #print("The best parameters are %s with a score of %0.2f" % (classifier.best_params_, classifier.best_score_))

cv_params = {'n_estimators': [100,200,300,400,500], 'learning_rate': [0.01, 0.1]}
other_params = {'learning_rate': 0.1,  'n_estimators': 100, 'max_depth': 5, 'min_child_weight': 1, 'seed': 27, 'nthread': 6,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
                'objective': 'multi:softmax', 'num_class': 5}
model = XGBClassifier(**other_params)
classifier = GridSearchCV(estimator=model, param_grid=cv_params, cv=3, verbose=1, n_jobs=6)
classifier.fit(train_feature, train_label)
print("The best parameters are %s with a score of %0.2f" % (classifier.best_params_, classifier.best_score_))


print('\npredict for test data.')
#test_feature = lle.transform(test_data)
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




