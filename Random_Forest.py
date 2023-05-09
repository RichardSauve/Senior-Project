from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_excel('TOR.xlsx', header=0)
data = data.set_index('GP', drop=True)
X = data.drop(['Date', 'AT', 'Opponent', 'GF', 'GA', 'OT', 'Result', 'CF', "CA", "FF", "FA", "FOW", "FOL","PDO"], axis=1).to_numpy()
y = data['Result'].to_numpy()
win = y == 'W'
y = np.where(win, 1, 0)

X_train = X[:int(len(X)*6.5/10)]
X_test = X[int(len(X)*6.5/10):]

y_train = y[:int(len(X)*6.5/10)]
y_test = y[int(len(X)*6.5/10):]
#
# hyper = []
# auc_scores = []
# auc_scoresT = []
# n_estimators = np.arange(100, 1500, 50)
# for i in n_estimators:
#     model = RandomForestClassifier(n_estimators=350, max_leaf_nodes=2)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     predictionsT = model.predict(X_train)
#     hyper.append(i)
#     auc_scores.append(accuracy_score(y_test, predictions))
#     auc_scoresT.append(accuracy_score(y_train, predictionsT))
#
# plt.plot(hyper, auc_scores, color='orange', label='Test Set')
# plt.plot(hyper, auc_scoresT, color='blue', label='Train Set')
# plt.xlabel('n_estimators')
# plt.ylabel('Accuracy Score')
# plt.legend()
# plt.tight_layout()
# plt.savefig('n_estimators3')
# plt.show()

model = RandomForestClassifier(n_estimators=350, max_leaf_nodes=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# #
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='g')
plt.title("TOR Confusion Matrix")
plt.xlabel('Predicted Vales')
plt.ylabel("Actual Values")
plt.savefig('Confusion Matrix TOR')
plt.show()
#
# features = data.columns[[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22, 25, 26]]
# print(features)
# importances = model.feature_importances_
# indices = np.argsort(importances)
#
# plt.figure(1)
# plt.title('Feature Importances TOR')
# plt.barh(range(len(indices)), importances[indices], color='blue', align='center')
# plt.yticks(range(len(indices)), features[indices])
# plt.xlabel('Relative Importance')
# plt.savefig('Feature Importance TOR2')
# plt.show()
from sklearn import tree
#
# fn=features
# cn= ['Team Lose', 'Team Win']
# fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize = (4,4), dpi=800)
# tree.plot_tree(model.estimators_[10],
#                feature_names = fn,
#                class_names=cn,
#                filled = True);
# fig.savefig('rf_individualtree.png')
