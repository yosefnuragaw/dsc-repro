from copy import deepcopy
from typing import Dict, Tuple
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

from DSC.phase import load_dataset_and_model, evaluate_model, evaluate_consortia, evaluate_benchmark
from DSC.utils import merge_data, exclude_data, pad_array , read_all_config
from DSC.model.catboost import *
import yaml

class DSCPlatform():
    def __init__(self, config_path: str, dataset_path:str, robustness: str = "greedy" , verbose: bool = True):
        if robustness == "greedy" or robustness =="breadth":
            self.robustness = robustness
        else:
            raise Exception("ERROR \t Unknown parameter value!")
        
        self.config_path = config_path
        self.verbose = verbose
        self.__initiate_consortium(config_path)
        

    def install_data(self):
        s = time.time()
        for i in range(self.participants):
            data_path = self.data.get("DATASET", {})[i]
            benchmark_path = self.data.get("BENCHMARK", {})[i]
            data_i, model_i, self.cat = load_dataset_and_model(
                dataset_path = f"dataset\{data_path}", 
                target = self.target, 
                task = self.task)
            benc_i, p_i = evaluate_model(
                benchmark_path = f"dataset\{benchmark_path}",
                model =  model_i, 
                target = self.target, 
                task = self.task)
            
            self.models[i]= model_i
            self.dataset[i]=data_i
            self.benchmark[i]=benc_i
            self.P[i]=p_i
            if self.verbose:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARTICIPANT {i} \t PAYOFF:{p_i}")
        runtime = time.time() - s
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SETUP \t\t RUNTIME: {runtime:.2f}s.")
        return runtime

    def allocate_revenue(self) :
        s = time.time()
        tmp_ds = deepcopy(self.dataset)
        tmp_bm = deepcopy(self.benchmark)  
        
        (y,y_pred),p_ip,_ = evaluate_consortia(
                tuning_data = merge_data(tmp_ds),
                benchmark_data = merge_data(tmp_bm),
                task = self.task, 
                categorical= self.cat)

        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CONSORTIUM \t PAYOFF:{p_ip}")

        Q = len(y)
        M = np.zeros((Q, self.participants + 1), dtype=int)

        if self.robustness == "greedy":
            bad_data = []
            stable = False
            while not stable:
                banned_in_this_round = False

                for i in list(tmp_ds.keys()):
                    ds_no_i = exclude_data(tmp_ds, i)
                    bm_no_i = exclude_data(tmp_bm, i)
                    bad_data.append(i)
                    (y, y_pred), new_pip, _ = evaluate_consortia(
                        tuning_data=merge_data(ds_no_i),
                        benchmark_data=merge_data(bm_no_i),
                        task=self.task,
                        categorical=self.cat
                    )

                    M[:, i] = pad_array((y_pred == y).astype(int), self.benchmark, bad_data)

                    if new_pip > p_ip:
                        if self.verbose:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARTICIPANT {i} \t DOWNGRADE PAYOFF DIFF:{p_ip - new_pip:.5f}")
                        
                        del tmp_ds[i]
                        del tmp_bm[i]
                        banned_in_this_round = True
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] REMOVE \t\t PARTICIPANT {i}")
                        p_ip = new_pip
                        break  
                    else:
                        bad_data.remove(i)

                if not banned_in_this_round or len(tmp_bm) == 1:
                    stable = True
        else:   
            bad_data = []  
            while True:
                worst_pip, p = 0, None
                for i in list(tmp_ds.keys()):

                    ds_no_i = exclude_data(tmp_ds, i)
                    bm_no_i = exclude_data(tmp_bm, i)

                    (y,y_pred),new_pip,_ = evaluate_consortia(
                        tuning_data = merge_data(ds_no_i),
                        benchmark_data = merge_data(bm_no_i),
                        task = self.task, 
                        categorical= self.cat)
                    M[:, i] = pad_array((y_pred == y).astype(int),self.benchmark,bad_data+[i])

                    if new_pip > p_ip:
                        if self.verbose:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PARTICIPANT {i}  \t DOWNGRADE PAYOFF DIFF:{p_ip - new_pip:.5f}")
                        if new_pip > worst_pip:
                            worst_pip = new_pip
                            p = i

                if p == None:
                    break
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] REMOVE \t\t PARTICIPANT {p}")
                    bad_data.append(p)
                    p_ip = worst_pip
                    del tmp_ds[p]
                    del tmp_bm[p]

                    if len(tmp_bm) == 1:
                        break
                    

        self.cleaned_dataset = tmp_ds
        self.cleaned_benchmark = tmp_bm

        (y,y_pred),p_ip,_ = evaluate_consortia(
            tuning_data = merge_data(self.cleaned_dataset),
            benchmark_data = merge_data(self.cleaned_benchmark ),
            task = self.task, 
            categorical= self.cat)
        
        M[:, -1] = pad_array((y_pred == y).astype(int),self.benchmark,list(set(self.benchmark) - set(self.cleaned_benchmark)))

        entitlements = {i: 0.0 for i in list(self.dataset.keys())}
        total = 0.0  
        
        for idx in range(Q):
            if M[idx, -1] == 1:  
                total += 1  #
                contributor = []
                for id in range(self.participants):
                    if M[idx, id] == 0:
                        contributor.append(id)
                if contributor:
                    share = 1.0 / len(contributor)
                    for id in contributor:
                        entitlements[id] += share

        if total > 0:
            for id in entitlements:
                entitlements[id] = (entitlements[id] / total) * 100


        self.cleaned_dataset = tmp_ds
        self.cleaned_benchmark = tmp_bm

        runtime = time.time() - s
        if self.verbose:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALLOCATION \t RUNTIME: {runtime:.2f}s.")
        return entitlements,runtime
    
    def evaluate_robustness(self) -> Dict:
        tmp_d = []
        tmp_c = []

        _,p_ip_raw, ml_i = evaluate_consortia(
            tuning_data = merge_data(self.dataset),
            benchmark_data = merge_data(self.benchmark),
            task = self.task, 
            categorical= self.cat)
        
        _,p_ip_cleaned,_ = evaluate_consortia(
            tuning_data = merge_data(self.cleaned_dataset),
            benchmark_data = merge_data(self.cleaned_benchmark),
            task = self.task, 
            categorical= self.cat)

        for id in range(self.participants):
            _, d_i = evaluate_benchmark(
                benchmark_data= self.benchmark[id],
                task=self.task,
                model= ml_i
            )
                
            _, dp_i = evaluate_benchmark(
                benchmark_data= merge_data(self.benchmark),
                task=self.task,
                model= self.models[id]
            )

            _, rdp_i = evaluate_benchmark(
                benchmark_data= merge_data(self.cleaned_benchmark),
                task=self.task,
                model= self.models[id]
            )
            tmp_c.append(d_i) 
            tmp_d.append(rdp_i-dp_i)

        return {
            "A": p_ip_cleaned-p_ip_raw,
            "B": p_ip_raw - np.array(list(self.P.values())),
            "C": p_ip_cleaned - np.array(tmp_c),
            "D": np.array(tmp_d)
        }

    def __initiate_consortium(self, path:str):
        with open(path, "r") as file:
            self.data = yaml.safe_load(file)
            self.target = self.data.get("TARGET","")
            self.task = self.data.get("TASK","")
            self.participants = self.data.get("PARTICIPANTS", 0)
        
        self.dataset = {}
        self.models = {}
        self.benchmark = {}
        self.P = {}
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INITIATING \t CONFIG: {path}")


    def evaluate_runtime(self)->Dict[int,List]:
        res = {}
        configs = read_all_config()
        for conf in configs:
            self.__initiate_consortium(conf)
            t_install = self.install_data()
            _,t_alloc = self.allocate_revenue()
            res[self.participants] = [t_install,t_alloc]
        return res
            
                
