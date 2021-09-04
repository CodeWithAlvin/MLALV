# necessary imports
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import mean_squared_error
from math import sqrt 

def prepare_report(y_true,y_pred,type,log=True,avg="micro"):
    """
    type ==> classification / regression
    analyze the classification model on the parameters
    f1-score , recall , precision , accuracy

    analyze the regression model on the parameters
    mse and rmse
    """

    if type == "classification":
        f1 = f1_score(y_true,y_pred, average=avg)# analysing f1 score   
        precision = precision_score(y_true, y_pred, average=avg)# analysing precision score
        recall = recall_score(y_true, y_pred, average=avg)# analysing recall score
        accuracy = accuracy_score(y_true, y_pred)# analysing accuracy score                
        if log:
            print(f"f1-score : {f1}\nprecision : {precision}\nrecall : {recall}\naccuracy : {accuracy}")
        return {"f1" : f1, "precision" : precision, "recall" : recall, "accuracy" : accuracy}
    
    elif type == "regression":   
        mse = mean_squared_error(y_true,y_pred)
        rmse = sqrt(mse)
        if log:
            print(f" mse : {mse} \n rmse : {rmse}")
        return {"mse":mse , "rmse":rmse}
    
    else:
        raise ValueError(f"unknown value({type}) must be either `classification` or `regression`.")


