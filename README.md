# IEEE-CIS Fraud Detection

Imagine standing at the check-out counter at the grocery store with a long line behind you and the cashier not-so-quietly announces that your card has been declined.
In this moment, you probably aren’t thinking about the data science that determined your fate. Embarrassed, and certain you have the funds to cover everything needed 
for an epic nacho party for 50 of your closest friends, you try your card again. Same result. As you step aside and allow the cashier to tend to the next customer,
you receive a text message from your bank. “Press 1 if you really tried to spend $500 on cheddar cheese.”. While perhaps cumbersome (and often embarrassing) in the moment,
this fraud prevention system is actually saving consumers millions of dollars per year.

The task in this project therefore is to classify whether a transaction is fraudulent given a large-scale dataset. The data comes from Vesta's real-world e-commerce transactions 
and contains a wide range of features from device type to product features. If successful, you’ll improve the efficacy of fraudulent transaction alerts for millions of people
around the world, helping hundreds of thousands of businesses reduce their fraud loss and increase their revenue.
And of course, you will save party people just like you the hassle of false positives.

## Installation
### Downloading the Data
- Clone this repository to your computer
- Navigate to the project directory `cd packages/random_forest_model/random_forest_model from your terminal`
- Run `mkdir input`
- Use `cd input` to go into the directory where the data is to be stored
- Download the data files from Kaggle
  - [Click here](https://www.kaggle.com/c/ieee-fraud-detection/data) to go straight to the Data page on Kaggle
  - If you don't have a Kaggle account then you'd have to create one

### Installing the Requirements 
- Navigate to the Random Forest Package using `cd packages/random_forest_model` 
- Install the requirements using `pip install -r requirements`
To install requirements for the API: 
- Navigate to the API using `cd packages/ml_api`
- Install the requirements using `pip install -r requirements`
