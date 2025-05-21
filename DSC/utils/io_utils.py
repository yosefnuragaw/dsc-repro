import pandas as pd
import numpy as np
from typing import Tuple,List
import os

def read_df(df: pd.DataFrame, target_col: str)-> Tuple[np.ndarray,np.ndarray]:
    cat = []

    try:
        X = df.drop(columns=target_col)
        y = df[target_col]
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col].fillna(X[col].median(), inplace=True)
            else:
                cat.append(X.columns.get_loc(col)) 
                X[col].fillna(X[col].mode()[0], inplace=True)

        
        return X.values,y.values,cat
    
    except Exception as e:
        raise Exception("Read Dataframe Error!") from e
    
def read_all_config(directory_path:str = "DSC\\config") -> List:
    res = []
    config_path = os.path.join(directory_path )
    if os.path.exists(config_path):
        for filename in os.listdir(config_path):
            if os.path.isfile(os.path.join(config_path, filename)):
                res.append(os.path.join(config_path, filename))
    return res