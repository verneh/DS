import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('churn.csv')
# including the countries gave a higher score for churners   0.74      0.49      0.59      0.87
# excluding the countries gave this score for churners       0.70      0.43      0.54      0.85
# "gender", "hascrcard", "isactivemember"                  0.63      0.41      0.50      0.84
data = data.drop(["Unnamed: 0"], axis=1)
#
x = data.drop('exited', axis=1).values
y = data['exited'].values

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

ep = RandomForestClassifier(bootstrap=False, max_depth=40, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=1400)

ep.fit(x_train, x_test)
results = ep.predict(y_train)

# for checking purposes.
# target_names =['no churn','churn']

# st.write("Accuracy:",metrics.accuracy_score(y_test, results))
# st.write("Class Report:", metrics.classification_report(y_test, results, target_names=target_names))

# c = metrics.confusion_matrix(y_test, results)
# st.write(c)

# sve this as a pickle file.
filename = 'rfc_model.pkl'
joblib.dump(ep, filename)
