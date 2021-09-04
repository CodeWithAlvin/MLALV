# necessary imports
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class SimpleTransformer:
    def __init__(self, encoder="ordinal", strategy="median"):
        """
        encoder ==> ordinal / onehot
        strategy ==> median / mean / most_frequent / constant
        
        Do:
            ~ imputing remove/replace  nan none
            ~ Scaling scale the given data
            ~ Encoding encode the objexts or string present in data
        
        """
        #specifing available encoders
        available_encoders={
            "ordinal" : OrdinalEncoder,
        	"onehot" : OneHotEncoder,
        }

        #raising error if encoder value is not identified
        if encoder not in available_encoders.keys() :
            raise ValueError(f"unknown encoder {encoder}. encoder must be any of {available_encoders.keys()}")
        
        #setting some parameters 
        self.encoder = available_encoders[encoder]
        self.strategy = strategy
        

    def fit(self,data):
        """
        create  piplines 
        merge them 
        fit on the training data
        make pipeline available using self
        """
        #pipeline for handling numeric data remove na's and then scale the data 
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy=self.strategy)),
            ('std_scaler', StandardScaler()),
        ])

        #checking which encoder is specified and setting them in pipeline
        # dome seperately bcz of d/f parameters of onehot encoder and ordinal encoder
        if self.encoder == OneHotEncoder:
            obj_pipe = Pipeline([
                ("encoder",self.encoder(handle_unknown='ignore'))
            ])

        else:
            obj_pipe = Pipeline([
                ("encoder",self.encoder())
            ])

        # getting all the columns in dataframe
        _columns = list(data.columns)
        
        # seprating all the columns on basis of wheater they contain numeric data or objects
        cat_attribs = [col for col in _columns if data[col].dtype == object]
        num_attribs = [col for col in _columns if col not in cat_attribs]
        
        # merging both numeric and obj pipeline in one.
        self.full_pipeline = ColumnTransformer([
            ("num", num_pipe, num_attribs),
            ("cat", obj_pipe, cat_attribs),
        ])

        # fitting the data given
        self.full_pipeline.fit(data)

    def transform(self,data):
        """
        Do transformation on data and return it .
        """
        return self.full_pipeline.transform(data)
