import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

def predict():
    #Prepare training data
    df = pd.read_csv('./data/abalone.data',
        names=('sex', 'length', 'diameter', 'height', 'weight', 'age'),
        usecols=[0,1,2,3,4,8])
    df['age'] = df['age'] + 1.5
    df_sex = pd.get_dummies(df['sex'], prefix='sex')
    train_data = df_sex.join(df).drop(['sex'], axis=1)
    X_train = train_data.iloc[:, :7]
    y_train = train_data.iloc[:, 7]

    #Train the model for prediction
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    #Do prediction by using randomforest model which was trained above
    prediction = model.predict(X_train)

    #Calculate MSE, and it shows accuracy of prediction
    mean_squared_error(y_train, prediction)

    #Dump trained random forest model into local file so that it can be reused by other case
    joblib.dump(model, './dump/abalone_randomforest.pkl')
    y_train_prediction = np.array([y_train, prediction])

    #Dump prediction data and actual data into local file
    y_train_prediction = np.array([y_train, prediction])
    np.save('./dump/y_train_prediction', y_train_prediction)

if __name__ == "__main__":
    predict()
