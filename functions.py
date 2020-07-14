import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import roc_curve, auc 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import itertools
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

""" This function calculates the confusion
    matrix of a classifier model"""
def confusion(cnf_matrix,target):
  plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) 

# Add title and axis labels
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

# Add appropriate axis scales
  class_names = set(target) # Get class labels to add to matrix
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

# Add labels to each cell
  thresh = cnf_matrix.max() / 2. # Used for text coloring below
# Here we iterate through the confusion matrix and append labels to our visualization 
  for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
    plt.text(j, i, cnf_matrix[i, j],
             horizontalalignment='center',
             color='white' if cnf_matrix[i, j] > thresh else 'black')

# Add a legend
  plt.colorbar()
  plt.show()


"""This is a function that calculates class imbalance"""
def class_imbalance(X_train, X_test, y_train, y_test):
  weights = [None, 'balanced', {1:2, 0:1}, {1:10, 0:1}, {1:100, 0:1}, {1:1000, 0:1}]
  names = ['None', 'Balanced', '2 to 1', '10 to 1', '100 to 1', '1000 to 1']
  colors = sns.color_palette('Set2')

  plt.figure(figsize=(10,8))
  for n, weight in enumerate(weights):
    # Fit a model
    logreg = LogisticRegression(fit_intercept=False, C=100, class_weight=weight, solver='liblinear', random_state = 42)
    model_log = logreg.fit(X_train, y_train)
    print(model_log)

    # Predict
    y_hat_test = logreg.predict(X_test)

    y_score = logreg.fit(X_train, y_train).decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
    print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
    print('-------------------------------------------------------------------------------------')
    lw = 2
    plt.plot(fpr, tpr, color=colors[n],
             lw=lw, label='ROC curve {}'.format(names[n]))

  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])

  plt.yticks([i/20.0 for i in range(21)])
  plt.xticks([i/20.0 for i in range(21)])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic (ROC) Curve')
  plt.legend(loc='lower right')
  plt.show()
 


"""This function implements SMOTE, balancing act for class imbalances"""
def smote_class(X_train, X_test, y_train, y_test):
  ratios = [0.1, 0.25, 0.33, 0.5, 0.7, 1]
  names = ['0.1', '0.25', '0.33','0.5','0.7','even']
  colors = sns.color_palette('Set2')

  plt.figure(figsize=(10, 8))

  for n, ratio in enumerate(ratios):
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train) 
    logreg = LogisticRegression(fit_intercept=False, C=100, solver ='liblinear')
    model_log = logreg.fit(X_train_resampled, y_train_resampled)
    print(model_log)

    # Predict
    y_hat_test = logreg.predict(X_test)

    y_score = logreg.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
    print('AUC for {}: {}'.format(names[n], auc(fpr, tpr)))
    print('-------------------------------------------------------------------------------------')
    lw = 2
    plt.plot(fpr, tpr, color=colors[n],
             lw=lw, label='ROC curve {}'.format(names[n]))

  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.yticks([i/20.0 for i in range(21)])
  plt.xticks([i/20.0 for i in range(21)])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic (ROC) Curve')
  plt.legend(loc='lower right')
  plt.show()
    
    
"""This function plots the features that drives the 
    perfomance of the model"""
def plot_feature_importances(model, X_train, predictors):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), predictors.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    

    
"""This function outputs the evaluation metrics for each 
   machine learning model"""

def evaluation_metrics(y_train,y_hat_train,y_test,y_hat_test):
    print('Training Precision: ', precision_score(y_train, y_hat_train))
    print('Testing Precision: ', precision_score(y_test, y_hat_test))
    print('\n')
    print('Training Recall: ', recall_score(y_train, y_hat_train))
    print('Testing Recall: ', recall_score(y_test, y_hat_test))
    print('\n')
    print('Training Accuracy: ', accuracy_score(y_train, y_hat_train))
    print('Testing Accuracy: ', accuracy_score(y_test, y_hat_test))
    print('\n')
    print('Training F1-Score: ', f1_score(y_train, y_hat_train))
    print('Testing F1-Score: ', f1_score(y_test, y_hat_test))
    

    
    
def feature_engine(data):
    row = data[0]
    transform_data = pd.DataFrame(row).transpose()
    transform_data.columns = ['total_items', 'total_sales', 'discounted_sales',
                     'browsing_duration', 'age', 'household_income', 'loyalty_points','month', 'loyalty_card', 'education', 'marital_status', 'region', 'gender']
    
    non_normal = ['total_items', 'total_sales', 'discounted_sales',
                     'browsing_duration', 'age', 'household_income', 'loyalty_points']
    
    data_cat = transform_data[['month', 'loyalty_card', 'education', 'marital_status', 'region', 'gender']]
    data_cont = transform_data[non_normal].astype(float)
    
 
    
    
        
    return data_cont,data_cat
 
    
    
    
    
                                   
                                   
                                   
                                   
          
    


    
    
   
    
    
    
    