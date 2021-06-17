import numpy as np 
import pandas as pd

class Process:
    def __init__(self):
        pass
    
    @staticmethod
    def fullstack(data, encoder="onehot", strategy="median"):
        """
        preprocess the data for ml algorithm 
        changes all the string to integer,
        fill all the Na, nan, none ,
        scale the data.
        
        inputs ::
            data --> to perform operations
            strategy --> strategy for SimpleImputer            make_copy --> True > alter copy data else alter original data
            encoder --> ordinal || onehot <-- sklearn encoders
        output ::
            altered data
            encoder object for getting encoder values
        """
        # necessary imports
        from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
        from sklearn.impute import SimpleImputer
        # code goes here
        
        # converting into dataframe if it is not
        _data = pd.DataFrame(data)
        
        # getting all the columns in dataframe
        _columns = list(_data.columns)
        
        # seprating all the columns on basis of wheater they contain numeric data or objects
        objects = [col for col in _columns if data[col].dtype == object]
        numeric = [col for col in _columns if col not in objects]
        
        # performing encoding of string data
        # all the encoders will be in this dictionary
        availble_encoders = {
        	"ordinal" : OrdinalEncoder,
        	"onehot" : OneHotEncoder,
        	}
        	       	     	               	
        # transforming string data and making changing to data
        if encoder not in availble_encoders.keys() :
            raise ValueError(f"unknown encoder {encoder}. encoder must be any of {availble_encoders.keys()}")
        
        _encoder = availble_encoders[encoder]()        
        if encoder == "onehot" and objects :
            _data[objects] = _encoder.fit_transform(_data[objects].astype(str)).toarray()
        elif objects:
            _data[objects] = _encoder.fit_transform(_data[objects].astype(str))

        # running Simple Imputer on numeric data to fill all na and nan's
        _imputer = SimpleImputer(strategy=strategy)
        _data[numeric] = _imputer.fit_transform(_data[numeric])
        
        # running Standard Scalar on all data
        _scale = StandardScaler()
        _data[numeric] = _scale.fit_transform(_data[numeric])
        
        return _data,_encoder
        
        


