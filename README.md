# PJ Morgan Bank Loan Data Analysis

![alt text](Intro_Image "Loan Analysis")

## Summary

Supervised learning algorithms were deployed in a data-set containing the information of 300K loans including important metrics that help us better understand the process by which future loans can be measured against, with the end goal of minimizing delinquent loans and the loss of the principal amount loaned


## Business Problem 

The costliest mistake the bank can make is to issue a loan that will end up in default or deliquency. The aim of this model is to understand the metrics we need to montior though out the lifecyle of a loan so as to avoid any potential loss of the principal amount on top of interest not accurred. 

This kind of error in our model is a false negative error and is signifiactnly more costly than a false positive error in which the bank would only lose hypothetical interest amount as the loan was never distributed. 

## Data

The data used in these supervised learning algorithmic models came from 300,000 records with a ratio or 2:1, not delinquent - to - delinquent. The features implemented in this model are the following:

Loan Amount

Funded Amount

Funded Amount Invested

Interest Rate

Installments

Annaul Income

FICO Range – Low

FICO Range – High

Total Payment

Total Payment – Investement

Total Recovered Principal

Total Recovered Interest

Last FICO Range – High

Last FICO Rango  - Low 


## Results

### Logistic Regression Model


The first model is a Logistic Regression Model. This model allows us to view the precission by which our model can detect False Negative and False Positive errors



![alt text](Confusion_Matrix "Confusion Matrix")


### Random Forests Model


Our second model was initiated by decision tree as a preliminary step to our Random Forest algorithm.

After we run our Random Forest we are able to interpret the importance of each indivial feature and verify the precission of our model 







![alt text](Feature_Importance "Feature Importance")






![alt text](Classification_Report "Classification Report")



We can see the importnace of monitoring FICO scores throught the lifecylce of the loan. We can also see the model preformed at a 97% precission rate on our test data. 

It is imperative to gage the FICO fluctuation of the applicant from 36 -60 months prior to the loan application in order as this is our most important feature and our best predictor as to whether an applicant will become delinquent in the future or not.


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
