import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


pd.set_option('display.max_columns', None)

df = pd.read_csv("data.csv", sep=",")
y = df['Status']

df.drop('Status', axis=1, inplace=True)
df.drop('ID', axis=1, inplace=True)
df.drop('Recording', axis=1, inplace=True)

number_of_attributes = len(df.columns)


mms = MinMaxScaler()
df_norm = mms.fit_transform(df)

clf = SVC()
scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)

print(scores.mean())