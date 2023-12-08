![credit-risk-workflow-automation-768x512](https://github.com/njgeorge000158/credit-risk-classification/assets/137228821/03beeaa5-d44d-40bd-8ec8-6c716204189d)

----

# **Credit Risk Classification with Logistic Regression using Scikit-learn**

## Overview of the Analysis

The purpose of this analysis is to evaluate the performance of two supervised machine learning models for predicting consumer loan credit risk. These Logistic Regression models use a data set of historical activity from a peer-to-peer lending services company to identify the creditworthiness of borrowers. Specifically, the models predict whether a loan is healthy or high-risk by analyzing the provided applicant information such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt.  Moreover, although the data set is comprehensive, there is significantly more information about healthy loans than high-risk loans (75,036 records vs. 2,500 records). With this in mind, the second model compensates for this inconsistency.

To accomplish the analysis, the machine learning process includes the following steps:

1. Separate the data into a features variable, X, and a labels variable, y.
2. Check the labels variable's value count.
3. Further split the features and labels variables into training and testing data sets.
4. Fit a Logistic Regression model by using the training data.
5. Evaluate the model’s performance using the testing data to find the accuracy, precision, and recall scores.
8. Resample the data using the Scikit-learn function, RandomOverSampler, to address the labels's value count imbalance.
9. Check the labels variable's value count again.
10. Create and fit another Logistic Regression model with the resampled data.
11. Evaluate the performance of the resampled model using the same metrics.

## Results

- Machine Learning Model 1: Logistic Regression with Original Data
* Overall Accuracy: 95.2%
* Balanced Accuracy: 99%
* Precision (Health Loans): 100%
* Precision (High-risk Loans): 85%
* Recall (Healthy Loans): 99%
* Recall (High-risk Loans): 99%

<img width="483" alt="Screenshot 2023-12-08 at 1 52 28 PM" src="https://github.com/njgeorge000158/credit-risk-classification/assets/137228821/aa699f09-e45a-4030-acd7-de834576c262">

- Machine Learning Model 2: Logistic Regression with Resampled Data 
* Overall Accuracy: 99.4%
* Balanced Accuracy: 99%
* Precision (Health Loans): 100%
* Precision (High-risk Loans): 84%
* Recall (Healthy Loans): 99%
* Recall (High-risk Loans): 91%

<img width="491" alt="Screenshot 2023-12-08 at 2 12 14 PM" src="https://github.com/njgeorge000158/credit-risk-classification/assets/137228821/3ba7b976-5cac-4154-9247-5a2dec85c37d">

## Summary

The first Logistic Regression model does an excellent job predicting healthy loans with a small number of false positives and negatives leading to a precision score of 100%, a recall score of 99%, and an f1-score of 100%.  Nevertheless, this model less accurately predicts high-risk loans with a precision of 85%, a recall of 91%, and an f1-score of 88%. The balanced accuracy, 99%, is higher than the actual accuracy, 95%, because of the significant difference in labels's value counts. The first model's potential for an increase in accuracy and the comparatively inadequate performance in predicting high-risk loans vs. health loans are concerning. Thus, the model warrants further optimization either by closing the value count gap with additional data or random oversampling: the second model uses the latter to solve this problem.

In terms of accuracy, the second Logistic Regression model with random oversampling matches the first model for predicting healthy loans and outperforms it for high-risk loans. For instance, the number of accepted healthy loans falls (18,663 to 18,649); the number of rejected high-risk loans expands (563 to 615); the number of false positives increases slightly (102 to 116); and the number of false negatives significantly drops (56 to 4). Moreover, using random oversampling to generate additional synthetic samples for the minority label class eliminates the labels's value count discrepancy leading to, among other things, the balanced accuracy score matching the overall accuracy score, 99%. For healthy loans, both models have 100% precision, 99% recall, and 100% f1-scores; for high-risk loans, although the precision, 85%, declines by 1% to 84%, the recall, 92%, increases by 8% to 99%, and the f1-score, 88%, increases by 3% to 91%. Consequently, using random oversampling with the Logistic Regression model maintains its identification of healthy loans while improving its identification of high-risk loans.

Ultimately, the second Logistic Regression model trained with resampled data demonstrates a significant improvement in predicting high-risk loans while maintaining comparable performance for healthy loans. Consequently, I recommend it as a better choice for credit risk analysis.
