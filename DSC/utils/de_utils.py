from typing import List, Tuple, Dict
import numpy as np

def exclude_data(data: Dict[int, Tuple[np.ndarray,np.ndarray]],i: int)->Dict[int, Tuple[np.ndarray,np.ndarray]]:
     if len(data.items()) == 1:
          raise Exception("All data removed")
     
     return {key: value for key, value in data.items() if key != i}

def merge_data(data: Dict[int, Tuple[np.ndarray,np.ndarray]])->Tuple[np.ndarray,np.ndarray]:
    X_merged = np.vstack([X for X, _ in data.values()])
    y_merged = np.hstack([y for _, y in data.values()])

    return X_merged, y_merged

def pad_array(raw: np.ndarray, reference: Dict[int, Tuple[np.ndarray,np.ndarray]], i:List[int]):
    # _, y = merge_data(reference)
    # total_samples = y.shape[0]
    # offset = 0
    # block_length = 0

    # for key in reference.keys():
    #     if key == i:
    #         block_length = reference[key][0].shape[0]
    #         break
    #     else:
    #         offset += reference[key][0].shape[0]

    # padded = np.empty(total_samples)
    # padded[:offset] = raw[:offset]
    # padded[offset:offset+block_length] = 0
    # padded[offset+block_length:] = raw[offset:]
    
    # return padded
    if not i:
        return raw
    
    _, y = merge_data(reference)
    total_samples = y.shape[0]
    
    padded = np.empty(total_samples)
    p_offset = 0
    c_offset = 0

    for key in reference.keys():
        block_length = reference[key][0].shape[0]
        
        if key in i:
            padded[p_offset:p_offset + block_length] = 0
        else:
            padded[p_offset:p_offset + block_length] = raw[c_offset:c_offset + block_length]
            c_offset += block_length

        p_offset += block_length
    return padded
