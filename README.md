# aapredictor
Demo project to predict the age of abalone by using random forest regressor.

## About
Use random forest regressor to predict the age of abalone.

## Running
This project requires Python 3 package.<BR>
First, install pipenv. Then:

```
git clone https://github.com/wanglianglin/aapredictor
cd aapredictor
pipenv install
pipenv run python predictionapp.py
```

Finally, below files will be save into *{data}* folder.

|File Name                 | Memo                                  |
|--------------------------|---------------------------------------|
|balone_randomforest.pkl   | trained model for prediction          |
|y_train_prediction.npy    | predict data and original actual data |
