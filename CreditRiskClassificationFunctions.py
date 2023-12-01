#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  CreditRiskClassificationFunctions.py
 #
 #  File Description:
 #      This Python script, CryptoClusteringFunctions.py, contains generic 
 #      Python functions for completing common tasks in the Credit Risk
 #      Classification Challenge.  Here is the list:
 #
 #      ModelPerformanceEvaluatorFunction
 #      
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  11/27/2023      Initial Development                     N. James George
 #
 #******************************************************************************************/

import PyConstants as constant
import PyLogSubRoutines as log_subroutine

import pandas as pd

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report


# In[2]:


CONSTANT_LOCAL_FILE_NAME \
    = 'CreditRiskClassificationFunctions.py'


# In[3]:


#*******************************************************************************************
 #
 #  Function Name:  ModelPerformanceEvaluatorFunction
 #
 #  Function Description:
 #      This function evaluates the logistic regression model and produces a confusion 
 #      matrix.
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  Series
 #          yTestSeries
 #                          The parameter is test data used for evaluation.
 #  Numpy Array
 #          predictionsNumpyArray
 #                          The parameter is predictions used for evaluation.
 #  Boolean
 #          originalFlagBoolean
 #                          The parameter indicates the type of data (original vs. 
 #                          random oversampling)
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  11/27/2023          Initial Development                         N. James George
 #
 #******************************************************************************************/

def ModelPerformanceEvaluatorFunction \
        (yTestSeries, 
         predictionsNumpyArray, 
         originalFlagBoolean = True):

    try:
        
        if originalFlagBoolean == True:
        
            dataTypeString ='Original'
    
        else:
        
            dataTypeString = 'Random Oversampling'
        
        
        confusionMatrixNumpyArray \
            = confusion_matrix \
                (yTestSeries, 
                 predictionsNumpyArray)
        
        indexStringList \
            = ['Actual Healthy', 
               'Actual High-Risk']
    
        columnStringList \
            = ['Predicted Healthy', 
               'Predicted High-Risk']
        
        targetNamesStringList \
            = ['healthy', 
               'high risk']
        

        accuracyScoreFloat \
            = balanced_accuracy_score \
                (yTestSeries, 
                 predictionsNumpyArray)
    
        confusionMatrixDataFrame \
            = pd.DataFrame \
                (confusionMatrixNumpyArray,
                 index = indexStringList,
                 columns = columnStringList)

        classificationReportString \
            = classification_report \
                (yTestSeries, 
                 predictionsNumpyArray, 
                 target_names = targetNamesStringList)
    
    
        log_subroutine \
            .PrintAndLogWriteText \
                ('\033[1m' 
                 + f'LOGISTIC REGRESSION MODEL ({dataTypeString})\n'
                 + '\033[0m\n'
                 + '1) '
                 + '\033[1m'
                 + 'Accuracy Score: '
                 + '\033[0m'
                 + f'{round(accuracyScoreFloat*100, 1)}%\n\n'
                 + '2) '
                 + '\033[1m'
                 + 'Confusion Matrix:\n'
                 + '\033[0m\n'
                 + f'{confusionMatrixDataFrame}\n\n'
                 + f'3) '
                 + '\033[1m'
                 + 'Classification Report:\n'
                 + '\033[0m\n'
                 + f'{classificationReportString}\n')
    
    
        return \
            accuracyScoreFloat, \
            confusionMatrixDataFrame, \
            classificationReportString
    
    except:
        
        log_subroutine \
            .PrintAndLogWriteText \
                (f'The function, ReturnOptimalKWithWCSSElbowFunction, '
                 + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
                 + f'was unable to return an optimal k value.')
    
        return \
            None


# In[ ]:




