import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

# 1. Read Data: For this assignment, assume input is CSV

def read_data(filename, index_col):
    df = pd.read_csv(filename, index_col = index_col)
    return df
# 2. Explore Data: You can use the code you wrote for assignment 1 here to generate distributions and data summaries
def explore_data(df):
    print("### Summary Statistics ###")
    print(df.describe())
    df.hist()
    plt.savefig('histograms')
    print("### Null Values ###")
    df.isnull().sum()

# 3. Pre-Process Data: Fill in misssing values
def fill_na_median(df, cols_to_fill=False, conditional_mean = False, group_col=None):
    #if no columns specified, fill all
    if not cols_to_fill:
        cols_to_fill = list(df.columns.values)
    if conditional_mean:
        if group_col == None:
            return "Cannot fill with conditional mean without a group_col specified"
        else:
            for col in cols_to_fill:
                df[col].fillna(df.groupby(group_col)[col].transform("mean"), inplace=True)
    else:
        for col in cols_to_fill:
            df[col].fillna(df[col].mean(), inplace=True)


# 4. Generate Features: Write a sample function that can discretize a continuous variable and 
# one function that can take a categorical variable and create binary variables from it.
# 5. Build Classifier: For this assignment, select any classifer you feel comfortable with 
# (Logistic Regression for example)

def log_reg(df, features, outcome_var):
    X = df[features]
    y = df[outcome_var]
    lgr = LogisticRegression()
    scores = cross_val_score(lgr, X, y, cv = 10)
    print(scores)
    print(scores.mean())

# 6. Evaluate Classifier: you can use any metric you choose for this assignment (accuracy is the easiest one)