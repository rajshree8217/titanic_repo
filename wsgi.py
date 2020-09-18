#!/usr/bin/env python
# coding: utf-8

# ####  Week 3 - Functions

# In[1]:


# Import Libraries
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Add in Week 3
# for graphical output to df.head and df.describe from within a function, use display NOT print
from IPython.display import display  


# Feature Engineering
import feature_engine.categorical_encoders as ce
import feature_engine.missing_data_imputers as mdi
from sklearn.preprocessing import MinMaxScaler

# Model and Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score,recall_score


# In[2]:


# Run environment Setup
import warnings
warnings.simplefilter("ignore")

#  numpy print options
# used fixed dpoint notation for floats with 4 decimals
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

# Display options on terminal for pandas dataframes
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# global variables available to all functions in this python file
TRAINED_MODEL = 0
MEDIAN_IMPUTER = 0
OHCE = 0
TO_SCALE =[]
SCALER = 0
TO_DROP = []


# #### Read data

# In[3]:


def read_data(filename):
    print("\n*****FUNCTION read_data*****")
    
    # Read the data file into a df
    df = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'],filename))
    
    # See the data in the df
    display(df.head())
    
    # Full data set Shape
    print("Shape of Full set:")
    print(df.shape)
       
    return(df)
# end of function read_data


# #### Data Exploration

# In[4]:


def disp_df_info(df):
    print("\n*****FUNCTION disp_df_info*****")
    
    # Create a Pie Chart to check Balance
    df['Survived'].value_counts(sort=True)

    #Plotting Parameters
    plt.figure(figsize=(5,5))
    sizes = df['Survived'].value_counts(sort=True)
    colors = ["grey", 'purple']
    labels = ['No', 'Yes']

    # Plot Pie chart
    plt.pie(sizes, colors = colors, labels = labels, autopct='%1.1f%%', shadow=True, startangle=270,)

    plt.title('Percentage of Churn in Dataset')
    plt.show()
    
    
    # display column Headers
    print("Column Headers:")
    print(df.columns)
          
    # print first 10 data samples
    print("Top 10 rows:")
    display(df.head(10))
    
    #Describe the df to check if features need scaling
    print("Statistics:")
    display(df.describe())
    
    # Identify the Categorical Vars and identify nulls
    print("Information:")
    print(df.info())
    
    # Count Nulls 
    print("Null Count:")
    print(df.isnull().sum())
    
    # Percent of Nulls
    print("Null Percent:") 
    print(df.isnull().mean())
    
    # Age has 20% Nulls - Plot Histogram of Age to see distribution in order to decide imputation method
    df['Age'].hist(bins=30)
    plt.title('Age Histogram')
    plt.xlabel('Age')
    plt.ylabel('Frequency/Count')
    plt.show()
    
    # Fare has 2% Nulls - Plot Histogram of Fare to see distributionin order to decide imputation method
    df['Fare'].hist(bins=30)
    plt.title('Fare Histogram')
    plt.xlabel('Fare')
    plt.ylabel('Frequency/Count')
    plt.show()
    
# end of function disp_df_info


# #### Data cleaning

# In[5]:


def data_cleaning(df_input):
    print("\n*****FUNCTION data_cleaning*****")
    
    df = df_input.copy(deep=True)
    
    # Print Shape
    print("Shape Before Dropping rows and columns:", df.shape)
    
    # Drop unwanted columns
    df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
    display(df.head(10))
    
    # Drop rows with Nulls using df.dropna(), will drop over 20% data
    # Embarked has 2 nulls, OK to drop rows with a low number of Nulls
    df = df[df['Embarked'].notnull()]
    print("Null Percent after dropping rows:") 
    print(df.isnull().mean())
      
    # Print Shape
    print("Shape After Dropping rows and columns:", df.shape)
    
    return(df)
    # end of functiom clean_data    


# #### Data Split into X/Feature and Y/target

# In[6]:


def data_split(df_input):
    print("\n*****FUNCTION data_split*****")
    
    df = df_input.copy(deep=True)
    
    # Create Y var
    y = df['Survived']
    print("Y/Target Var:")
    display(y.head(10))

    # Create X var
    x = df.drop(['Survived'], axis=1)
    print("X/Feature Var:")
    display(x.head(10))
    
    return(x,y)
# end of function data_split


# #### Feature Engineering

# In[7]:


def feature_engineering(x_input):
    print("\n*****FUNCTION feature_engineering*****")

    x = x_input.copy(deep=True)
    global MEDIAN_IMPUTER
    global OHCE
    
   
    MEDIAN_IMPUTER = mdi.MeanMedianImputer(imputation_method='median',
                                            variables=['Age','Fare'])
    
    MEDIAN_IMPUTER.fit(x)
    x=MEDIAN_IMPUTER.transform(x)
    print(MEDIAN_IMPUTER.imputer_dict_)
        
    OHCE=ce.OneHotCategoricalEncoder(variables=['Sex','Embarked'], 
                                                  drop_last=True)
        
    OHCE.fit(x)
    x=OHCE.transform(x) 
    print(OHCE.encoder_dict_)
        
    # Transformed df - No Nulls after imputation
    print("Null Count after Missing Data Imputation:")
    print(x.isnull().sum())

    # Transformed df - dummy vars created
    print("Dummy Variables after OHE:")
    display(x.head())

    return(x)
# end of feature_engineering function


# #### Feature Selection

# In[8]:


def feature_selection(x_input):
    print("\n*****FUNCTION feature_selection*****")
    
    x = x_input.copy(deep=True)

    global TO_DROP
    
    # Check the correlation of the variables
    corr_mat = x.corr()

    # Correlation Matrix visualized as Heatmap
    print("Correlation Martix for X/Feature Space:")
    plt.figure(figsize=(8,8))
    sns.heatmap(corr_mat, annot= True, cmap='coolwarm', center = 0 , vmin=-1, vmax=1)
    plt.show()

    #https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
    # Create correlation matrix
    corr_matrix = x.corr().abs()
   
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    print(upper)

    # Find index of feature columns with correlation greater than a user set value 
    TO_DROP = [column for column in upper.columns if any(upper[column] > 0.70)]
    print("Features to Drop:",TO_DROP)

    # Shape before dropping features
    print("Shape BEFORE Dropping features:", x.shape)

    # Drop features
    x.drop(x[TO_DROP], axis=1, inplace=True)

    # Shape after dropping features
    print("Shape AFTER Dropping features:", x.shape)

    return(x)
# end of feature_selection function


# #### Feature Scaling

# In[9]:


def feature_scaling(x_input):
    print("\n*****FUNCTION feature_scaling*****")

    global SCALER
    global TO_SCALE

    x = x_input.copy(deep=True)

    # Choose columns to scale, create a separate df
    TO_SCALE = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    dftoscale = x[TO_SCALE]
    
    # Call scaler to scale the df
    SCALER = MinMaxScaler()
    SCALER.fit(dftoscale)
    dftoscale = SCALER.transform(dftoscale)
    
    # put the scaled df back into original df
    x[TO_SCALE] = dftoscale
    
    return(x)
# end of feature_scaling function


# #### Model Fitting 

# In[10]:


def build_logreg_model(x_input, y_input):
    print("\n*****FUNCTION build_logreg_model*****")

    x = x_input.copy(deep=True)
    y = y_input.copy(deep=True)
    
      
    # Call Logistic Regession with no penalty
    mod = LogisticRegression(penalty='none')
    mod.fit(x,y)

    # Print the Intercept and the coef
    print('Intercept:', mod.intercept_)
    print('Coefficients:', mod.coef_)

    # Score the model
    score = mod.score(x, y)
    print('Accuracy Score:',score)

    # probability of being 0, 1 in binary clasification , threshold is .5
    y_prob=mod.predict_proba(x)
    print('Probabilities:',y_prob)
   
    # probability converted to predictions
    y_pred = mod.predict(x)
    print('Predictions:',y_pred)
   
    #################### Model Metrics ####################

    # Confusion Matrix gives the mistakes made by the classifier
    cm =confusion_matrix(y, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Confusion Matrix visualized
    plt.figure(figsize= (8,6))
    sns.heatmap(cm, annot= True, fmt= 'd', cmap = 'Reds')
    plt.title('Confusion Matrix Visualized')
    plt.xlabel('Predicted y_pred')
    plt.ylabel('Actuals / labels - y')
    plt.show()

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    # For Logistic Regression the model score is the Accuracy Ratio
    # (TP+TN)/(TP+TN+FP+FN)
    acc = accuracy_score(y,y_pred)
    print('Accuracy:',acc)

    # Precion = TP/(TP+FP)
    # Interpretation: out of all the predicted positive classes, how much we predicted correctly.
    pre = precision_score(y,y_pred)
    print('Precision:',pre)

    # Specificity = TN/(TN+FN)
    # Interpretation: out of all the -ve samples, how many was the classifier able to pick up
    spec = TN/(TN + FP)
    print('Specificity:',spec)
    
    # Recall/Sensitivity/tpr = TP/(TP+FN)
    # Interpretation: out of all the +ve samples, how many was the classifier able to pick up
    rec = recall_score(y,y_pred)
    tpr=rec
    print('Recall:',rec)

    # false positive rate(fpr) = FP/(FP + TN) = 1-specificity
    # Interpretation: False alarm rate
    fpr = FP/(FP + TN)
    print('False Positive Rate:',fpr)


    # Print Completion
    print('**************************** Model Ready to be used/invoked ***************************')

    # return the trained model
    return(mod,score)
# end of build_logreg_model function


# #### Week 4 - Flask

# In[11]:


# Import Flask 
from flask import Flask
from flask import render_template
from flask import request
from flask import send_file


# In[12]:


# import werkzeug to run your app as a web application
# from werkzeug.serving import run_simple


# In[13]:


# Create input file folder
upload_folder_name = 'input_titanic_folder'
upload_folder_path = os.path.join(os.getcwd(),upload_folder_name)
print('Upload folder path is:',upload_folder_path)
if not os.path.exists(upload_folder_path):
    os.mkdir(upload_folder_path)


# In[14]:


# Instantiate the Flask object 
application = Flask(__name__)
print('Flask object Instantiated')


# In[15]:


application.config['UPLOAD_FOLDER'] = upload_folder_path


# In[16]:


# home displays trainform.html
@application.route("/home", methods=['GET'])
def home():
    return render_template('trainform.html')
# end of home


# In[17]:


# submit on trainform.html
@application.route("/train", methods=['POST'])
def train():
    
    global TRAINED_MODEL
    
    file_obj = request.files.get('traindata')
    print("Type of the file is :", type(file_obj))
    name = file_obj.filename
    print(name)
    file_obj.save(os.path.join(application.config['UPLOAD_FOLDER'],name))
    
    # Is the File extension .csv
    if name.lower().endswith('.csv'):
        print('Input File extension good', name)
    else:
        print('***ERROR*** Input file extension NOT good')
        return render_template('trainform.html', errstr = "***ERROR*** Input file extension NOT good") 
    #End If
    
    # Steps to TRAIN the model
    titanic_df = read_data(name)
    disp_df_info(titanic_df)
    clean_df = data_cleaning(titanic_df)
    x,y=data_split(clean_df)
    x = feature_engineering(x)
    x = feature_selection(x)
    x = feature_scaling(x)
    TRAINED_MODEL,score = build_logreg_model(x,y)

    return render_template('trainresults.html',acc=score)
# end of home


# In[18]:


# Use model on trainresults.html
# OR Use model on predresults.html
@application.route("/predform", methods=['POST'])
def predform():
    return render_template('predform.html')
# end of home


# In[19]:


# submit on predform.html
@application.route("/make_pred", methods=['POST'])
def make_pred():
    
    global MEDIAN_IMPUTER
    
    file_obj = request.files.get('newdata')
    print("Type of the file is :", type(file_obj))
    name = file_obj.filename
    print(name)
    file_obj.save(os.path.join(application.config['UPLOAD_FOLDER'],name))
    
    # Is the File extension .csv
    if name.lower().endswith('.csv'):
        print('Input File extension good', name)
    else:
        print('***ERROR*** Input file extension NOT good')
        return render_template('predform.html', errstr = "***ERROR*** Input file extension NOT good") 
    #End If
    
    # Steps to USE model:
    # Call fx Read_data
    new_df = read_data(name)

    # Call fx data_cleaning
    clean_x = data_cleaning(new_df) 
    print('New Cleaned Data:',clean_x)
    
    # Feature Eng - Reuse the MEDIAN_IMPUTER
    print(MEDIAN_IMPUTER.imputer_dict_)
    new_x = MEDIAN_IMPUTER.transform(clean_x)
    print('New FE Data:',new_x)

    # Feature Eng - Reuse the OHCE
    print(OHCE.encoder_dict_)
    new_x = OHCE.transform(new_x)
    print('New FE Data:',new_x)

    #Feature Selection - Reuse TO_DROP
    #Drop the redundant features
    new_x.drop(new_x[TO_DROP], axis=1, inplace=True)
    print('New Selected Data:',new_x)

    # Feature Scale - Reuse SCALER, TO_SCALE
    dftoscale = new_x[TO_SCALE]

    # Call scaler to scale the df
    dftoscale = SCALER.transform(dftoscale)

    # put the scaled back into original
    new_x[TO_SCALE]=dftoscale
    print('New Scaled Data:', new_x)

    # Make Prediction - Reuse MODEL to make prediction
    new_pred = TRAINED_MODEL.predict(new_x)
    print('New Prediction:',new_pred)

    # new_pred is a np array in a row, transpose to column in order to join with original data frame
    new_pred = np.transpose(new_pred)

    # Add a new column to original data frame called 'Prediction' 
    # with the transposed new_pred np array
    new_df['Prediction']=new_pred


    # Save results to file on server without index
    new_df.to_csv(os.path.join(application.config['UPLOAD_FOLDER'],'result_'+ name),index=False)

    print("*************************** New Prediction Complete WITH FLASK ***************************************")

    # Return results to browser/client render_template OR send_file , http does NOT allow both.
    # return render_template('predresults.html',data=new_df)
    return(send_file(os.path.join(application.config['UPLOAD_FOLDER'],'result_'+ name),as_attachment=True))


# end of make_pred


# #### Main Program for Web App

# In[ ]:


# Main Program for Web app
# If __name__ = __main__ ,program is running standalone
if __name__ == "__main__":
    print("Python script is run standalone")
    print("Python special variable __name__ =", __name__)   
        
       
    # Run the flask app in jupyter noetbook needs run_simple 
    # Run the flask app in python script needs app.run
#     run_simple('localhost',5000, app, use_debugger=True)
    application.run('0.0.0.0',debug=True)

     
else:
    # __name__ will have the name of the module that imported this script
    print("Python script was imported")
    print("Python special variable __name__ =", __name__)   
#End Main program

