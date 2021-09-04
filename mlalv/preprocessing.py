# necessary imports
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class SimpleTransformer:
    def __init__(self, encoder="ordinal", strategy="median"):
        available_encoders={
            "ordinal" : OrdinalEncoder,
        	"onehot" : OneHotEncoder,
        }

        if encoder not in available_encoders.keys() :
            raise ValueError(f"unknown encoder {encoder}. encoder must be any of {available_encoders.keys()}")
        
        self.encoder = available_encoders[encoder]
        self.strategy = strategy
        

    def fit(self,data):
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy=self.strategy)),
            ('std_scaler', StandardScaler()),
        ])

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
        

        self.full_pipeline = ColumnTransformer([
            ("num", num_pipe, num_attribs),
            ("cat", obj_pipe, cat_attribs),
        ])

        self.full_pipeline.fit(data)

    def transform(self,data):
        return self.full_pipeline.transform(data)
