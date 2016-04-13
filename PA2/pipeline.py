import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

# 1. Read Data: For this assignment, assume input is CSV

def read_data(filename, index_col):
    '''
    Reads data from csv
    '''
    df = pd.read_csv(filename, index_col = index_col)
    return df

# 2. Explore Data: You can use the code you wrote for assignment 1 here to generate distributions and data summaries
def explore_data(df):
    print("### Summary Statistics ###")
    print(df.describe())
    df.hist()
    plt.savefig('histograms')
    print("### Null Values ###")
    print(df.isnull().sum())

def make_hist(df, col, num_bins=10):
    '''
    Generate a simple histogram

    df: DataFrame
    col: column to visualize
    num_bins: how many bins to have for the visualization
    '''
    print('Plotting ' + col)
    plt.hist(df[col], bins = num_bins)
    plt.title(col)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    filename = str(col) + '_hist'
    plt.savefig(filename)
    plt.close()

# 3. Pre-Process Data: Fill in misssing values
def fill_na_mean(df, cols_to_fill=False, conditional_mean = False, group_col=None):
    '''
    Fills missing values with mean or class conditional mean

    cols_to_fill: list of columns to impute values for. If none specified, fill all.
    conditional_mean: Boolean. If False, fills with unconditional mean
    group_col: Column to condition on. Required if conditional_mean is True
    '''
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

def discretize(df, col, bins, labels):
    '''
    Discretizes a continuous variable 

    Inputs:
        df: dataframe
        col: col name of variable to discretize
        bins: integer or list of cutoff points for different bins
        labels: False or list of names for each bin. must be same length as number of bins
    Modifies existing dataframe in place
    '''
    new_name = str(col) + '_disc'
    df[new_name] = pd.cut(df[col], bins, labels=labels)

def binarize(df, col):
    '''
    Takes a categorical variable and creates binary dummy variables from it
    Inputs:
        df: dataframe
        col: col name of variable to discretize
    
    Returns: new dataframe
    '''
    dummies = pd.get_dummies(df[col])
    df = df.join(dummies)
    return df
# 5. Build Classifier: For this assignment, select any classifer you feel comfortable with 
# (Logistic Regression for example)

def log_reg(df, features, outcome_var):
    '''
    Fits a logistic regression model, and scores it on accuracy
    Inputs:
        df: dataframe
        features: list of feature columns to include
        outcome_var: dependent variable name
    Outputs:
        scores: list of accuracy scores from cross validating model
        lgr: fitted model

    '''
    X = df[features]
    y = df[outcome_var]
    lgr = LogisticRegression()
    scores = cross_val_score(lgr, X, y, cv = 10)
    lgr.fit(X,y)
    print('Average accuracy for this model:')
    print(scores.mean())
    return scores, lgr

def predict_log_reg(test_df, features, lgr):
    '''
    Predict outcomes for test data and write to csv
    inputs:
        test_df: test DataFrame
        features: list of features to use
        lgr: fitted logistic regression model
    '''
    X = test_df[features]
    y = lgr.predict(X)
    index_list = list(test_df.index.values)
    y_df = pd.DataFrame(y, index=index_list, columns=['predicted'])
    final = pd.concat([test_df, y_df], axis=1)
    final.to_csv('predictions.csv', header=True)
