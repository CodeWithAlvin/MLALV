# necessary imports
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class Process:
    def __init__(self):
        pass
           
    @staticmethod  
    def _numeric_transform(data, strategy):
        """ 
         pipeline for numeric data
         SimpleImputer() --> filling na , nan's values
         StandardScaler() --> for Scaling data
        """  
        numeric_pipe = Pipeline([
        	("_imputer" , SimpleImputer(strategy = strategy) ),
        	("_scale" , StandardScaler() ),
        ])
        
        # calling pipeline to transform data
        return numeric_pipe.fit_transform(data)
    
    @staticmethod
    def _object_transform(data, encoder , available_encoders):
        """
        perform string encoding
        """
        object_pipe = Pipeline ([
        	("_encoder" , available_encoders[encoder]() ),
        ])      
        
        if encoder == "onehot" :
            raise ValueError("One hot encoder is currently not available we are working for it. use ordinal endcoder instead.")
            return object_pipe , object_pipe.fit_transform(data.astype(str)).toarray().tolist()
        else:
            return object_pipe , object_pipe.fit_transform(data.astype(str))

        
        
    @staticmethod
    def fullstack(data, encoder="ordinal", strategy="median",copy=True):
        """
        preprocess the data for ml algorithm 
        changes all the string to integer,
        fill all the Na, nan, none ,
        scale the data.
        
        inputs ::
            data --> to perform operations
            strategy --> strategy for SimpleImputer       
            encoder --> ordinal || onehot <-- sklearn encoders
            copy --> True || False , copy the data else alter original data
        output ::
            altered data
            encoder object for getting encoder values
        """        
        
        # all the available encoders will be in this dictionary
        available_encoders = {
        	"ordinal" : OrdinalEncoder,
        	"onehot" : OneHotEncoder,
        	}
        
        # checking if the given inputs are valid or not
        if encoder not in available_encoders.keys() :
            raise ValueError(f"unknown encoder {encoder}. encoder must be any of {available_encoders.keys()}")
        
        # handling copy of data
        if copy :
            _data = data.copy() #if copy = True then make a copy of data and operate on it
        else:
            _data = data # else operate on original data
                                  
        # getting all the columns in dataframe
        _columns = list(_data.columns)
        
        # seprating all the columns on basis of wheater they contain numeric data or objects
        objects = [col for col in _columns if data[col].dtype == object]
        numeric = [col for col in _columns if col not in objects]
                
        # performing numeric data operations
        if numeric :
            _data[numeric] = Process._numeric_transform(data[numeric] , strategy = strategy)
                
        # performing encoding of string data if there is any string data
        if objects :            
            _encoder , _data[objects] = Process._object_transform(data[objects] , encoder = encoder , available_encoders = available_encoders)                                    
            return _data , _encoder
       
        return _data,None
        