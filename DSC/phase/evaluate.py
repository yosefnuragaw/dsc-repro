import numpy as np
from typing import Tuple
import pandas as pd

from DSC.model.catboost import *
from DSC.utils import read_df

def evaluate_model(benchmark_path:str, model, target: str, task: str):
    X, y, cat = read_df(pd.read_csv(benchmark_path),target)
    _, p_i = evaluate(model, X, y, task)
    return (X,y), p_i

def evaluate_consortia(
        tuning_data: Tuple[np.ndarray,np.ndarray],
        benchmark_data:Tuple[np.ndarray,np.ndarray], 
        task: str, 
        categorical: List) -> Tuple[Tuple[np.ndarray,np.ndarray],float]:
    ml_ip = init_model(task=task, categorical=categorical)
    _, best_param = tune(
        model=ml_ip,
        X=tuning_data[0],
        y=tuning_data[1],
        scoring="accuracy" if task == "CLASSIFICATION" else "rmse"
        # scoring="f1_macro" if task == "CLASSIFICATION" else "rmse"
    )

    ml_ip.set_params(**best_param)
    ml_ip.fit(tuning_data[0], tuning_data[1])
    res, p_ip = evaluate(ml_ip, benchmark_data[0], benchmark_data[1], task)
    return res, p_ip, ml_ip

def evaluate_benchmark(
        benchmark_data:Tuple[np.ndarray,np.ndarray],
        task: str,
        model
        )-> Tuple[Tuple[np.ndarray,np.ndarray],float]:
    res, p_ip = evaluate(model, benchmark_data[0], benchmark_data[1], task)
    return res,p_ip
