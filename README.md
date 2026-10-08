# Advertising Sales Prediction using Linear Regression

A Machine Learning project that predicts sales using advertising data and Linear Regression.

## Description

This project uses the advertising dataset to predict **sales** based on:

- TV advertising
- Radio advertising
- Newspaper advertising

The project follows a complete Machine Learning pipeline including data loading, data cleaning, preprocessing, training, prediction, evaluation, and visualization.

## Dataset

The dataset contains the following columns:

- **TV** - TV advertising budget
- **radio** - Radio advertising budget
- **newspaper** - Newspaper advertising budget
- **sales** - Sales (Target variable)

## Machine Learning Type

**Supervised Learning - Regression**

## Algorithm

**Linear Regression**

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## ML Pipeline

1. Load the dataset
2. Remove unwanted columns
3. Check missing values
4. Display statistical summary
5. Check correlation between columns
6. Separate independent and dependent variables
7. Split data into training and testing sets
8. Create ML Pipeline
9. Train the Linear Regression model
10. Test the model
11. Evaluate the model using MSE, RMSE and R²
12. Calculate model coefficients and intercept
13. Compare actual and predicted sales
14. Plot actual vs predicted sales

## Input

```python
Advertising.csv
```

The model uses:

```text
TV
radio
newspaper
```

to predict:

```text
sales
```

## Output

The program displays:

```text
Mean Squared Error
Root Mean Squared Error
R Square value
Model coefficients
Intercept
Actual sale
Predicted sale
```

Example format:

```text
Actual sale    Predicted sale
-----------------------------
   16.9       16.408024
   22.4       20.889882
    21.4       21.553843
    7.3       10.608503
```

The program also displays a graph of **Actual Sales vs Predicted Sales**.

## Conclusion

The Linear Regression model learns the relationship between advertising expenditure and sales and predicts sales for unseen data.
