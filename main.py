import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer # for encoding categorical to numerical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # for splitting data into trains and tests
from sklearn.linear_model import LinearRegression # for training and predicting
from sklearn.preprocessing import PolynomialFeatures # for polynomial linear regression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def DTR():
    # importing data
    data = pd.read_csv('Position_Salaries.csv')
    # print(data.head(5))

    # independent variable
    x = data.iloc[:, 1:-1].values

    # dependent variable
    y = data.iloc[:, -1].values
    # print(x, y)

    # splitting into 4 parts
    # this time data set is too small only 10
    # skip

    # training dataset

    tree_reg = DecisionTreeRegressor(random_state=1) # no need parameter tuning in this time
    tree_reg.fit(x, y)

    # feature scaling
    # there is no equation and splits (different values come always)
    # so, there is no meaning to apply feature scaling for decision tree regression

    # prediction
    print('6.5 years of Salary is %d dollars' %tree_reg.predict([[6.5]]))
    # no transform inside of the predict() because it not feature scaling

    # Visualization the results with smoothness
    # Decision Tree should follow this side
    x_grid = np.arange(min(x), max(x), 0.1)
    x_grid = x_grid.reshape(len(x_grid), 1)
    plt.scatter(x, y, color='red', marker='X')
    plt.plot(x_grid, tree_reg.predict(x_grid), color='blue')
    plt.title('Decision Tree')
    plt.xlabel('Position')
    plt.ylabel('Salary')
    plt.show()

    # without smooth
    # completely useless
    plt.scatter(x, y, color='red', marker='X')
    plt.plot(x, tree_reg.predict(x), color='blue')
    plt.title('Decision Tree (No Smooth)')
    plt.xlabel('Position')
    plt.ylabel('Salary')
    plt.show()


if __name__ == '__main__':
    DTR()