import pandas as pd

from DSC.utils import read_df
from DSC.model.catboost import *

def load_dataset_and_model(dataset_path: str, target: str, task: str):
    X, y, cat = read_df(pd.read_csv(dataset_path),target)
    ml_i = init_model(task=task, categorical=cat)
    _, best_param = tune(
        model=ml_i,
        X=X,
        y=y,
        scoring="accuracy" if task == "CLASSIFICATION" else "rmse"
        # scoring="f1_macro" if task == "CLASSIFICATION" else "rmse"
    )

    ml_i.set_params(**best_param)
    ml_i.fit(X, y)
    return (X,y),ml_i,cat

