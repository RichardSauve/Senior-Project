import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_excel('games_data.xlsx')
data = data.dropna()
y = data['Win']
X = data[['PTS%', 'SRS', 'SOS', 'GF/G', 'GA/G', 'PP%', 'PK%', 'SV%', 'S%']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

nhl = pd.read_csv('nhl_elo.csv')
col = nhl.columns

nhl['date'] = pd.to_datetime(nhl['date'])
nhl = nhl.query('season == 2023')
col_to_keep = ['date', 'neutral', 'home_team','away_team','home_team_pregame_rating', 'away_team_pregame_rating',
               'home_team_winprob', 'away_team_winprob', 'overtime_prob','home_team_score', 'away_team_score',
               'home_team_postgame_rating', 'away_team_postgame_rating']

nhl = nhl[col_to_keep].reset_index(drop=True).dropna()

t = 0
p = 0
y_true = []
y_score = []
for i, r in nhl.iterrows():
    t = int((r['home_team_score'] - r['away_team_score']) > 0)
    p = r['home_team_winprob']
    y_true.append(t)
    y_score.append(p)

fpr, tpr, thresholds = roc_curve(y_true, y_score)
fpr1, tpr1, threholds1, = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)
roc_auc1 = auc(fpr1, tpr1)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr1, tpr1, color='blue', lw = 2, label='ROC curve (area = %0.2f)' % roc_auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()