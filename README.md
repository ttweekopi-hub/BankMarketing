# Bank Marketing Term Deposit EDA

Interactive Streamlit app: https://bankmarketing-gl8wwhxgwyxyfxa5xqutvi.streamlit.app/

## Overview

I built this Streamlit app to turn the Bank Marketing dataset into a guided exploratory data analysis story. The dataset describes a Portuguese bank's direct marketing campaigns, where clients were contacted by phone and the business outcome was whether each client subscribed to a term deposit.

The use case is a realistic marketing problem: calling every customer is expensive, repetitive, and often ineffective. A bank wants to understand which customers are more likely to subscribe, which campaign conditions are associated with better outcomes, and what data quality or modelling risks must be handled before building a predictive model.

## Business Context

Term deposits are an important retail banking product because they help banks secure customer funds for a fixed period. Marketing teams commonly use phone campaigns to promote these products, but response rates are usually low. In this dataset, only 4,640 out of 41,188 records resulted in a subscription, giving a positive class rate of about 11.27%.

That imbalance matters. A model that simply predicts "no" for every customer would appear highly accurate, but it would fail at the actual business goal: finding likely subscribers. This EDA therefore focuses on understanding the minority subscriber group and identifying practical signals that could support a better campaign targeting strategy.

## What The EDA Attempts To Solve

This project is not just about plotting charts. The EDA is designed to answer a few modelling and business questions:

1. What separates subscribers from non-subscribers?
2. Which customer, campaign, previous-contact, and economic variables appear useful?
3. Where are the data quality issues hidden?
4. Which features create leakage or misleading model performance?
5. What modelling strategy should come after the EDA?

## Dataset

The project uses the `bank-additional-full.csv` dataset, which contains:

- 41,188 rows
- 20 input features
- 1 binary target column: `y`
- Customer profile fields such as age, job, marital status, education, and loan status
- Campaign fields such as contact type, month, call duration, and number of contacts
- Previous campaign fields such as `pdays`, `previous`, and `poutcome`
- Economic context fields such as employment variation rate, consumer price index, Euribor 3-month rate, and number of employees

The target is whether the client subscribed to a term deposit.

## Streamlit App Sections

The app presents the EDA as a step-by-step narrative:

- **Overview**: introduces the dataset, target imbalance, and high-level story
- **Dataset Information**: explains feature groups, data types, and first rows
- **Data Quality and Target Distribution**: checks missingness, duplicates, hidden `unknown` labels, and class imbalance
- **Numerical Feature Analysis**: compares distributions such as `age`, `duration`, `campaign`, `pdays`, `previous`, `euribor3m`, and `nr.employed`
- **Categorical Feature Analysis**: reviews subscription rates by job, education, contact type, month, previous outcome, and other categories
- **Statistical Associations and Correlation Review**: uses chi-square tests, Mann-Whitney U tests, and correlation checks
- **Leakage and Modelling Risks**: highlights features that need careful treatment before modelling
- **Conclusions and Modelling Strategy**: converts EDA findings into a practical modelling plan
