import modelbit, sys
from typing import *
from sklearn.pipeline import Pipeline
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.ensemble._forest import RandomForestClassifier
import pandas as pd

clf_pipe = modelbit.load_value("data/clf_pipe.pkl") # Pipeline(steps=[('preprocess', ColumnTransformer(transformers=[('numerical', Pipeline(steps=[('num_impute', SimpleImputer())]), <sklearn.compose._column_transformer.make_column_selector object at 0x00...

# main function
def predict_survive(Pclass, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked):
    # Create a DataFrame from the input data
    data = pd.DataFrame({'Pclass': [Pclass],
                         'Sex': [Sex],
                         'Age': [Age],
                         'SibSp': [SibSp],
                         'Parch': [Parch],
                         'Ticket': [Ticket],
                         'Fare': [Fare],
                         'Cabin': [Cabin],
                         'Embarked': [Embarked]})

    # Make predictions
    prediction = clf_pipe.predict(data)

    return prediction[0]

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = predict_survive(...)
#   print(result)