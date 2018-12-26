# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HeartDisease.csv')

# Get the number of missing data points per column. This will show up in variable explorer
missing_values_count = dataset.isnull().sum()
print(missing_values_count)

#Taking care of missing values
dataset = dataset.dropna()

#import all the libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Check data type of data
print(dataset.info())


#Print the main matrices for the data
print(dataset.describe())

# Get the number of missing data points per column. This will show up in variable explorer
missing_values_count = dataset.isnull().sum()
print(missing_values_count)

#Plot only the values of num- the value to be predicted/Label
dataset["num"].value_counts().sort_index().plot.bar()

#Heat map to see the coreelation between variables, use annot if you want to see the values in the heatmap
plt.subplots(figsize=(12,8))
sns.heatmap(dataset.corr(),robust=True,annot=True)

#Detect outliers
plt.subplots(figsize=(15,6))
dataset.boxplot(patch_artist=True, sym="k.")

#Dependent and independent variable split
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 11].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Fitting XGBoost to the Training set
import xgboost as xgb
from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=1, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=5, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
classifier.fit(X_train, y_train)

# A parameter grid for XGBoost
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Printing the classification report
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test, y_pred))
print("Accuracy is " ,accuracy_score(y_test,y_pred)*100)


#Visualising the importance of attributes
xgb.plot_importance(classifier)

#Plotting ROC Curve and Calculating Area under ROC curve
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=0)

# Print ROC curve
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("ROC Curve")
plt.plot(tpr,fpr)
plt.show() 

# Print AUC
auc = np.trapz(fpr,tpr)
print('Area Under ROC Curve:', auc)


