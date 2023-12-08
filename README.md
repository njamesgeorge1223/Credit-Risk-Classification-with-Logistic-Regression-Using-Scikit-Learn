![credit-risk-workflow-automation-768x512](https://github.com/njgeorge000158/credit-risk-classification/assets/137228821/03beeaa5-d44d-40bd-8ec8-6c716204189d)

----

# **Credit Risk Classification with Logistic Regression using Scikit-learn**

## Overview of the Analysis

The purpose of this analysis is to evaluate the performance of two supervised machine learning models for predicting consumer loan credit risk. These Logistic Regression models use a data set of historical lending activity from a peer-to-peer lending services company to identify the creditworthiness of borrowers. Specifically, the models predict whether the loan is healthy or high-risk by analyzing the provided applicant information such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.  Moreover, although the data set is comprehensive, there is significantly more information about healthy loans than high-risk loans (75,036 records vs. 2,500 records). With this in mind, the second machine learning compensated for this discrepancy.

For the two models, the machine learning process included the following steps:

1. Separate the data into a features variable, X, and a labels variable, y.
2. Check the labels variable's value count.
3. Further split the features and labels variables into training and testing data sets.
4. Fit a Logistic Regression model by using the training data.
5. Evaluate the model’s performance using the testing data to find the accuracy, precision, and recall scores.
8. Resample the data using the Scikit-learn function, RandomOverSampler, to address the labels' value count imbalance.
9. Check the labels variable's value count again.
10. Create and fit another Logistic Regression model with the resampled data.
11. Evaluate the performance of the resampled model using the same metrics.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Logistic Regression with Original Data
  * Description of Model 1 Accuracy, Precision, and Recall scores.
 
   Accuracy: The overall accuracy of the model is 0.99, indicating that it correctly classifies 99% of the instances.
Precision:
 - Healthy loans (0): The model has a precision of 1.00, which means it's excellent at identifying true positives with very few false positives.
 - High-risk loans (1): The model has a precision of 0.87, indicating its moderate effectiveness in identifying high-risk loans with some false positives.
Recall:
 - Healthy loans (0): The model has a recall of 1.00, which means it's correctly identifying nearly all instances of healthy loans with very few false     negatives.
 - High-risk loans (1): The model has a recall of 0.89, indicating its effectiveness in identifying high-risk loans with some false negatives.

* Machine Learning Model 2: Logistic Regression with Resampled Data 
  * Description of Model 2 Accuracy, Precision, and Recall scores.

Accuracy: The overall accuracy of the model is 0.99, indicating that it correctly classifies 99% of the instances.
Precision:
 - Healthy loans (0): The model has a precision of 0.99, which means it's excellent at identifying true positives with very few false positives.
 - High-risk loans (1): The model also has a precision of 0.99, indicating its effectiveness in identifying high-risk loans with very few false positives.
Recall:
 - Healthy loans (0): The model has a recall of 0.99, which means it's correctly identifying nearly all instances of healthy loans with very few false  negatives.
 - High-risk loans (1): The model has a recall of 0.99, indicating its effectiveness in identifying high-risk loans with very few false negatives.



## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.



Based on the results, the logistic regression model trained with resampled data (Model 2) performs better than the model trained with original data (Model 1), particularly in predicting high-risk loans. Model 2 demonstrates higher precision and recall scores for high-risk loans, which is crucial in minimizing potential financial losses for the lending company.

I recommend using the logistic regression model trained with resampled data (Model 2) for credit risk analysis, as it shows a significant improvement in predicting high-risk loans compared to the original model. This model will help the company effectively assess loan applications and make informed decisions when approving or rejecting loans, thus mitigating credit risk.
