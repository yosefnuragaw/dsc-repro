from catboost import Pool,CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from catboost.utils import get_gpu_device_count

import numpy as np 
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from typing import List, Tuple
SEED = 43

def init_model(task: str,categorical:List[int]):
    """
    Method to initialize catboost model
    """

    if task == "REGRESSION":
        return CatBoostRegressor(random_seed=SEED,cat_features = categorical,verbose=0)
    elif task == "CLASSIFICATION":
        return CatBoostClassifier(random_seed=SEED,cat_features = categorical,verbose=0)
    else:
        raise ValueError("Unknown task!")
    
def tune(model: CatBoostClassifier | CatBoostRegressor, X: np.ndarray, y: np.ndarray, scoring: str):
    """
    Method to hyperparameter tuning catboost model using grid-search
    """
    # Simplicity for faster testing (Note: add more parameter based on preference)
    param_grid = {
        "learning_rate": [0.05, 0.1],      
        'iterations': [100],
        
        # "learning_rate": [0.05, 0.1],  
        # 'iterations': [100,200],
        # "depth": [4, 8],                  
          
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, verbose=0,n_jobs=-1)
    # grid_search = GridSearchCV(model, param_grid, cv=10, scoring=scoring, verbose=0, n_jobs=-1)
    grid_search.fit(X,y)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate(model: CatBoostClassifier | CatBoostRegressor, X:np.ndarray,y:np.ndarray, task: str) -> Tuple[Tuple[np.ndarray,np.ndarray],float]:
    res = []

    if task == "REGRESSION":
        scoring_func = mean_squared_error

    elif task == "CLASSIFICATION":
        scoring_func = accuracy_score
        # scoring_func = f1_score
    else:
        raise ValueError("Unknown task!")

    y_te_pred = model.predict(X)
    # perf_score = scoring_func(y, y_te_pred,average="macro")
    perf_score = scoring_func(y, y_te_pred)
    
    return (y,y_te_pred) , perf_score