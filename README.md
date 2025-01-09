# Customer Churn Prediction
## Overview

This project aims to predict customer churn using machine learning techniques. Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company. Predicting churn is crucial for businesses as it allows them to take proactive measures to retain valuable customers.

## Data Preparation

1. **Loaded Data:**
   - Imported the dataset `customer_churn_data.csv` using Pandas:
     ```python
     import pandas as pd
     df = pd.read_csv("Churn_Modelling.csv")
     ```

2. **Inspected Dataset:**
   - Used `df.info()`, and `df.head()` to examine data structure, dimensions, and the first few records.

3. **Removal of unwanted column:**
   - Removed:
     ```python
     df.drop(['RowNumber'],axis=1,inplace=True)
     ```

---

## Exploratory Data Analysis (EDA)

The EDA phase involved investigating the dataset to gain insights into customer churn patterns. I utilized visualizations and summary statistics to understand the data better:

### Summary Statistics

- **`df.describe()`:** This function provided descriptive statistics for numerical features like 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'. It helped understand the distribution, central tendency, and dispersion of these features.
- **Observations:** By examining the summary statistics, I gained insights into the typical customer profile and identified potential areas of interest for further investigation.

### Churn Distribution by Geography

- **Bar Plot:** I used a bar plot to visualize the mean churn rate across different geographical locations (France, Spain, Germany). This allowed us to identify any geographical variations in churn behavior.
- **Observations:** The bar plot revealed that Germany had the highest churn rate compared to France and Spain, indicating a potential geographical influence on churn.

### Churn Distribution by Gender

- **Bar Plot:** I used a bar plot to visualize the mean churn rate based on gender (Female, Male). This helped us understand if gender played a role in customer churn.
- **Observations:** The bar plot indicated a higher churn rate for female customers compared to male customers, suggesting a potential gender-based difference in churn behavior.

### Credit Score and Churn Relationship

- **Histogram:** I used a histogram to visualize the distribution of credit scores for churned and non-churned customers. This helped us understand if there was a relationship betIen credit score and churn.
- **Observations:** The histogram shoId that churned customers tended to have lower credit scores on average compared to non-churned customers, suggesting a potential link betIen creditworthiness and churn.

### Credit Score Group and Churn Relationship

- **Count Plot:** I used a count plot to visualize the number of churned and non-churned customers within different credit score groups (Low, Medium, High). This helped us further analyze the relationship betIen credit score and churn.
- **Observations:** The count plot revealed that customers with lower credit scores are more likely to churn compared to those with higher credit scores, reinforcing the potential link betIen creditworthiness and churn.

### Estimated Salary, Balance, and Churn Relationship

- **Bar Plots:** I used bar plots to compare the estimated salary and balance for churned and non-churned customers, further exploring potential factors influencing churn.
- **Observations:** The bar plots indicated that while estimated salary might have some relationship with churn, balance did not show a significant association with churn behavior.

### Correlation Matrix

- **Heatmap:** I used a heatmap to visualize the correlations betIen numerical features. This helped us identify potential relationships and multicollinearity betIen variables.
- **Observations:** By examining the heatmap, I identified correlations betIen features such as age and churn, balance and number of products, and geographical location with balance. This information guided our feature selection and model building process.

## Model Building and Evaluation

After performing EDA and feature engineering, I proceeded to build and evaluate machine learning models to predict customer churn. Here's a summary of the process:

### Data Splitting

- **`train_test_split`:** I used this function from scikit-learn to split the dataset into training and testing sets. I allocated 80% of the data for training and 20% for testing. This split ensured that I could evaluate the model's performance on unseen data.

### Model Selection and Training

- **Model Selection:** I chose a variety of classification models to experiment with, including:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Naive Bayes
- **Model Training:** I trained each model using the training data (`x_train`, `y_train`) to learn patterns and relationships betIen the features and the target variable (churn).

### Model Evaluation

- **Performance Metrics:** I evaluated the performance of each model using the following metrics:
    - **Accuracy:** The proportion of correctly classified instances.
    - **R-squared (for Logistic Regression):** A measure of how Ill the model explains the variance in the target variable.
    - **Mean Squared Error (MSE) (for Logistic Regression):** The average squared difference betIen predicted and actual values.
- **Evaluation Process:** I used the testing data (`x_test`, `y_test`) to make predictions with each trained model and compared the predictions to the actual churn values to calculate the performance metrics.

### Model Comparison and Selection

- **Visualization:** I visualized the performance of different models using a bar plot to compare their accuracy scores.
- **Selection:** Based on the evaluation results and the bar plot, I identified the Random Forest and Support Vector Machine models as the top performers, achieving the highest accuracy scores.

## Key Findings

The analysis and modeling process yielded several important insights into customer churn:

**1. Geography Matters:** 
- Customers located in Germany exhibited a higher churn rate compared to those in France and Spain. This suggests that geographical factors may influence customer behavior and churn likelihood.

**2. Gender Differences:**
- Female customers Ire found to have a higher churn rate than male customers. This indicates a potential gender-based difference in churn patterns that warrants further investigation.

**3. Credit Score Correlation:**
- A negative correlation was observed betIen credit score and churn. Customers with loIr credit scores Ire more likely to churn, suggesting that creditworthiness may be a significant factor in customer retention.

**4. Credit Score Groups:**
- Customers categorized into the "Low" credit score group displayed a higher propensity to churn compared to those in the "Medium" and "High" credit score groups. This further emphasizes the importance of credit score in predicting churn.

**5. Estimated Salary and Balance:**
- While estimated salary might have a slight relationship with churn, the balance in customer accounts did not show a strong association with churn behavior. This implies that factors other than account balance may play a more significant role in customer decisions to churn.

**6. Model Performance:**
- Among the models evaluated, Random Forest and Support Vector Machine (SVM) achieved the highest accuracy scores in predicting customer churn. This suggests that these models are Ill-suited for identifying customers at risk of churning.

**7. Business Implications:**
- These findings provide valuable insights for businesses to develop targeted strategies for customer retention. By focusing on specific customer segments, such as those with loIr credit scores or located in Germany, businesses can proactively address potential churn risks.

## Conclusion

This project demonstrates the effectiveness of machine learning in predicting customer churn. The insights gained from this analysis can be used by businesses to implement strategies for customer retention and reduce churn rates.

## Future Work

Further improvements can be made by exploring more advanced machine learning techniques, incorporating external data sources, and fine-tuning model parameters.