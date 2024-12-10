# import dependencies
import pandas as pd
import sklearn as sk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time

# I'm breaking this into sections for a jupyter notebook
# before main function, make sure to define functions below
def main():
  # load cleaned data from csv
  X, y = load_data()
  print(X.head())
  
  # way too much data to do a pairplot
  # sns.pairplot(phishing_data.iloc[0:1000, :], kind= 'kde', height=0.5, aspect=2.25)
  
  vif_results = show_vif(X, y)
  print(vif_results)

  # drop columns based on results of vif
  params_to_drop = 5
  X = drop_high_variance_columns(X, params_to_drop, vif_results)

  # this will be used for tracking efficiency by parameter count
  parameter_count = len(X.columns)
  # (optional, consider regression after for basic model)

  # start out with manually changin the amount of parameters, and recording how long it took (I'll make a dict that auto adds new counts)
  svc_parameter_test_results = {}

  train_X, train_y, test_X, test_y = train_test_split(X, y, test_size=0.2)
  # build an SVM
  # Time how long it takes with various parameters removed to make a case for the improvement due to VIF
  # and show the test results due to different parameter usage.
  start_time = time.perf_counter()
  svc = SVC(kernel='rbf')
  C_range = np.logspace(-5, 5, 11, base=2)
  gamma_range = np.logspace(-5, 5, 11, base=2)
  param_grid = {"C": C_range, "gamma": gamma_range}
  svc_clf = GridSearchCV(svc, param_grid)
  svc_clf.fit(train_X, train_y)
  end_time = time.perf_counter()
  time_to_train = end_time - start_time

  svc_parameter_test_results["{}".format(X.columns): {"{} parameters".format(len(X.columns)): time_to_train}]



  # build a random forest
  # Time how long it takes with various parameters removed to make a case for the improvement due to VIF
  # and show the test results due to different parameter usage.
  rf_parameter_test_results = {}

  start_time = time.perf_counter()
  rfc = RandomForestClassifier()
  n_learners_range = np.arange(10, 210, 50)
  param_grid = {'n_learners': n_learners_range}
  rfc_clf = GridSearchCV(rfc, param_grid)
  rfc_clf.fit(train_X, train_y)
  end_time = time.perf_counter()
  time_to_train = end_time - start_time

  rf_parameter_test_results["{}".format(X.columns): {"{} parameters".format(len(X.columns)): time_to_train}]
  # validate on test data

  # print nice graph of train and test performance
  # I want graphs for:
  # parameter selection from grid search + test/train performance
  svc_train_results = svc_clf.cv_results_
  rf_train_results = rfc_clf.cv_results_

  # test results
  svc_test_results = svc_clf.score(test_X, test_y)
  rf_test_results = rfc_clf.score(test_X, test_y)


  # comparison of two models optimally for test data

  


def load_data():
  file_path = "PhiUSIIL_Phishing_URL_Dataset.csv"
  df = pd.read_csv(file_path)
  # drop filename because it's okay to drop according to source
  df.drop(['FILENAME'], axis=1, inplace=True)
  # drop URL, domain, and Title because it's unclear how they'd help, they're unique for each row.
  df.drop(['URL', 'Domain', 'Title'], axis=1, inplace=True)
  # encode TLD
  le = LabelEncoder()
  # Allison alluded to that in the post you linked, recommending use of the most frequent category as the reference when performing that type of VIF calculation.
  df_enc = le.fit_transform(df['TLD'].values)

  df_enc = pd.DataFrame(df_enc, columns=['TLD_encoded'])
  X = df.drop(['label', 'TLD'], axis=1)
  X = pd.concat([X, df_enc], axis=1)
  y = df['label']

  return X, y

def show_vif(X, y):
  vif_data = pd.DataFrame()
  vif_data["Feature"] = X.columns
  vif_data['VIF'] = [vif(X.values, i) for i in range(len(X.columns))]
  return vif_data

def drop_high_variance_columns(X, n, results):
  # make sure to not drop the categorical variables encoded
  # The variables with high VIFs are indicator (dummy) variables that represent a categorical variable with three or more categories. 
  # If the proportion of cases in the reference category is small, 
  # the indicator variables will necessarily have high VIFs, 
  # even if the categorical variable is not associated with other variables in the regression model.
  print("Dropping the column named: {}".format(results.columns[49]))
  print("hopefully it's the encoded column")
  results.drop(50, axis = 0)
  results = results.sort_values('VIF', axis = 0)
  n_worst_parameters = results['Feature'][-n:]
  return X.drop(n_worst_parameters, axis = 1)

if __name__ == "__main__": 
  main()