
#importing library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Funtion dividing data into training and testing
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 0:29] 
    Y = balance_data.values[:, 29] 
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 

# Function to perform Random Over-Sampling Example Technique (ROSE)
def rose_tech(X,Y):
    
    ros = RandomOverSampler()
    X_ros, Y_ros = ros.fit_sample(X, Y)
    return X_ros, Y_ros
      
# Function to perform Synthetic Minority Oversampling Technique (SMOTE)
def smote_tech(X,Y):
    
    smote = SMOTE(ratio='minority')
    X_sm, Y_sm = smote.fit_sample(X, Y)
    return X_sm, Y_sm



def transform(X_train):
    X_train_MinMax = MinMaxScaler().fit_transform(X_train)
    X_train_Scaler = StandardScaler().fit_transform(X_train)
    X_train_MaxAbs =  MaxAbsScaler().fit_transform(X_train)
    X_train_Normalizer = Normalizer().fit_transform(X_train)
    return X_train_MinMax,X_train_Scaler, X_train_MaxAbs, X_train_Normalizer
    
    
# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=10, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 10, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 

# Function to perform Logistic Regression
def logistic(X_train, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg
    
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: \n", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : \n", 
    classification_report(y_test, y_pred)) 
    
  
# Function to perform Random Forest 
def rand_forest(X_train, y_train):

    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(X_train, y_train)
    
    return rf

# Function to perform Support Vector Machines (SVM)
def SVM(X_train, y_train, X_test, y_test):
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)  
    y_pred = svclassifier.predict(X_test)
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred)*100)
    print("Report: \n", classification_report(y_test,y_pred))
    print("Accuracy: ",accuracy_score(y_test, y_pred)*100)
    return y_pred
        
# Driver code 
def main(): 
      
    
    #reading csv into pandas dataframe
    print("Reading data...")
    df = pd.read_csv("creditcard.csv", na_values = 'N/A')

    #deleting column "Time"
    df.drop("Time", axis = 1, inplace = True)
    
    #displaying count of Fraudulent cases vs non fraudulent
    df1 = (df.groupby("Class", axis = 0).Class.agg('count')).to_frame()
    df1.columns = ['Count']
    df1.reset_index(inplace= True)
    print("Target value count matrix on original dataset: \n", df1)
    
    #displaying dataframe with NaN's
    #df[pd.isnull(df).any(axis=1)]

    # Building Phase 
    # Splitting Data into training & testing calling a function
    print("Spliting data into Training and Testing with a ratio of 70:30...")
    X, Y, X_train, X_test, y_train, y_test = splitdataset(df) 
    
    # Apply transformation techniques
    X_train_MinMax,X_train_Scaler, X_train_MaxAbs, X_train_Normalizer = transform(X_train)
    
    # ROSE technique
    print('Implementing Random Over-Sampling Technique')
    X_train_rose, y_train_rose = rose_tech(X_train,y_train)
    print('After ROSE model, ',X_train_rose.shape[0] - X.shape[0], 'new random points were generated')
    
    # SMOTE technique
    print('Implementing Synthetic Minority Over Sampling Technique')
    X_train_smote, y_train_smote = smote_tech(X_train, y_train)
    print ('After SMOTE model, ',X_train_smote.shape[0] - X.shape[0], 'new random points were generated')
    
    # Deriving correlation between attributes before preprocessing and displaying graphically
    corr = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(df.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.show()
    
    # Displaying a table view of correlation matrix with absolute value of correlation greater than 0.3
    corr1 = corr.stack().reset_index()
    corr1.columns = ['a', 'b', 'c']
    corr2 = corr2 = corr1[(abs(corr1.c) > 0.3) & (abs(corr1.c) <1.0)].sort_values(by = 'c')
    corr2.drop_duplicates(subset='c', inplace= True)
    
    
    # Decision Tree using original dataset
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = train_using_entropy(X_train, X_test, y_train)
    clf_gini_rose = train_using_gini(X_train_rose, X_test, y_train_rose)
    clf_entropy_rose = train_using_entropy(X_train_rose, X_test, y_train_rose)
    clf_gini_smote = train_using_gini(X_train_smote, X_test, y_train_smote)
    clf_entropy_smote = train_using_entropy(X_train_smote, X_test, y_train_smote)
    
    # Decision Tree using preprossed data
    print("Implementing Decision Tree (GINI) on MinMax preprocessed data")
    clf_gini_MinMax = train_using_gini(X_train_MinMax, X_test_MinMax, y_train)
    y_pred_gini_MinMax = prediction(X_test_MinMax, clf_gini_MinMax) 
    cal_accuracy(y_test, y_pred_gini_MinMax)
    
    print("Implementing Decision Tree (Entropy) on MinMax preprocessed data")
    clf_entropy_MinMax = train_using_entropy(X_train_MinMax, X_test_MinMax, y_train)
    y_pred_entropy_MinMax = prediction(X_test_MinMax, clf_entropy_MinMax) 
    cal_accuracy(y_test, y_pred_entropy_MinMax)
    
    print("Implementing Decision Tree (GINI) on MaxAbs preprocessed data")
    clf_gini_MaxAbs = train_using_gini(X_train_MaxAbs, X_test_MaxAbs, y_train)
    y_pred_gini_MaxAbs = prediction(X_test_MaxAbs, clf_gini_MaxAbs) 
    cal_accuracy(y_test, y_pred_gini_MaxAbs)
    
    print("Implementing Decision Tree (Entropy) on MaxAbs preprocessed data")
    clf_entropy_MaxAbs = train_using_entropy(X_train_MaxAbs, X_test_MaxAbs, y_train)
    y_pred_entropy_MaxAbs = prediction(X_test_MaxAbs, clf_entropy_MaxAbs) 
    cal_accuracy(y_test, y_pred_entropy_MaxAbs)
    
    print("Implementing Decision Tree (GINI) on Scaler preprocessed data")
    clf_gini_Scaler = train_using_gini(X_train_Scaler, X_test_Scaler, y_train)
    y_pred_gini_Scaler = prediction(X_test_Scaler, clf_gini_Scaler) 
    cal_accuracy(y_test, y_pred_gini_Scaler)
    
    print("Implementing Decision Tree (Entropy) on Scaler preprocessed data")
    clf_entropy_Scaler = train_using_entropy(X_train_Scaler, X_test_Scaler, y_train)
    y_pred_entropy_Scaler = prediction(X_test_Scaler, clf_entropy_Scaler) 
    cal_accuracy(y_test, y_pred_entropy_Scaler)
    
    print("Implementing Decision Tree (GINI) on Normalizer preprocessed data")
    clf_gini_Normalizer = train_using_gini(X_train_Normalizer, X_test_Normalizer, y_train)
    y_pred_gini_Normalizer = prediction(X_test_Normalizer, clf_gini_Normalizer) 
    cal_accuracy(y_test, y_pred_gini_Normalizer)
    
    print("Implementing Decision Tree (Entropy) on Normalizer preprocessed data")
    clf_entropy_Normalizer = train_using_entropy(X_train_Normalizer, X_test_Normalizer, y_train)
    y_pred_entropy_Normalizer = prediction(X_test_Normalizer, clf_entropy_Normalizer) 
    cal_accuracy(y_test, y_pred_entropy_Normalizer)
    
    # Operational Phase 
    print("Results Using Gini Index on original dataset:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy on original dataset:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    print("Results Using Gini Index on ROSE data:") 
      
    # Prediction using gini 
    y_pred_gini_rose = prediction(X_test, clf_gini_rose) 
    cal_accuracy(y_test, y_pred_gini_rose)
    
    print("Results Using Entropy on ROSE data:") 
    # Prediction using entropy 
    y_pred_entropy_rose = prediction(X_test, clf_entropy_rose) 
    cal_accuracy(y_test, y_pred_entropy_rose)
    
    print("Results Using Gini Index on SMOTE data:")   
    # Prediction using gini 
    y_pred_gini_smote = prediction(X_test, clf_gini_smote) 
    cal_accuracy(y_test, y_pred_gini_smote)
    
    print("Results Using Entropy on SMOTE data:") 
    # Prediction using entropy 
    y_pred_entropy_smote = prediction(X_test, clf_entropy_smote) 
    cal_accuracy(y_test, y_pred_entropy_smote)
    
    # Building logistic model
    print("Implementing Logistic Regression...")
    logreg_model = logistic(X_train, y_train)
    
    # Predicting results of Logistic Model
    print("Results Using Logistic Regression on Original dataset:")
    y_pred_logistic = prediction(X_test, logreg_model)
    cal_accuracy(y_test, y_pred_logistic)
    
    # Random Forest
    clf_random = rand_forest(X_train, y_train)
    
    
    # SVM using original dataset
    print("Implementing SVM on original data set")
    SVM(X_train, y_train, X_test, y_test)
    
    # SVM using ROSE data
    print("Implementing SVM on ROSE data set")
    SVM(X_train_rose, y_train_rose, X_test, y_test)
    
    # SVM using SMOTE data
    print("Implementing SVM on SMOTE data set")
    SVM(X_train_smote, y_train_smote, X_test, y_test)
    
    
    
    # Calling main function 
if __name__=="__main__": 
    main() 
    
