import numpy as np
import random
import copy
import scipy.stats as ss
from tqdm import tqdm

def rk_key_eff(rank_array, key):
    return np.argsort(np.argsort(rank_array, axis=1), axis=1)[:, key]

def group_traces(traces, plaintexts):
    container = np.zeros((256, traces.shape[1]))
    for i in range(256):
        index = np.where(plaintexts==i)[0]
        container[i] = np.mean(traces[index], axis=0)
    return container

def rk_key_eff(rank_array, key):
    return np.argsort(np.argsort(rank_array, axis=1), axis=1)[:, key]

def corr2_coeff_rowwise2(A, B):
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    ssA = np.einsum('ij,ij->i',A_mA,A_mA)
    ssB = np.einsum('ij,ij->i',B_mB,B_mB)
    return np.einsum('ij,ij->i',A_mA,B_mB)/np.sqrt(ssA*ssB)

def min_max(x, min=0, max=1):
    X_std = (x - np.min(x)) / (np.max(x)-np.min(x))
    X_scaled = X_std * (max - min) + min
    return X_scaled

def get_prediction_table(TA_prediction, plaintext, all_keys):
    num_tasks = plaintext.shape[1]
    key_container = np.zeros((num_tasks, num_tasks, 256))
    rank_container = np.zeros((num_tasks, num_tasks))
    for a in range(num_tasks):
        for b in range(a+1, num_tasks):
            # print(a, b)
            # Perform attack
            # calculate averaged prediction based on each plaintext
            grouped_pred_1 = group_traces(TA_prediction[a, :], plaintext[:, a])
            grouped_pred_2 = group_traces(TA_prediction[b, :], plaintext[:, b])
            grouped_pred_1 = ss.rankdata(grouped_pred_1, method='dense', axis=-1) - 1
            grouped_pred_2 = ss.rankdata(grouped_pred_2, method='dense', axis=-1) - 1  
            for k in range(256):
                conv_pred_1 = grouped_pred_1[np.transpose([range(grouped_pred_1.shape[0])]), np.bitwise_xor([range(256)],np.full((grouped_pred_1.shape[0], 1), k))]   
                corr_list = corr2_coeff_rowwise2(conv_pred_1, grouped_pred_2[np.bitwise_xor(range(256),k)])
                key_container[a, b, k] = np.mean(corr_list)
                key_container[b, a, k] = key_container[a, b, k]
            rank_container[a, b] = 255-rk_key_eff([key_container[a, b]], all_keys[a]^all_keys[b])
            rank_container[b, a] = rank_container[a, b]
            # print("Final rank for differential attack: ", rank_evol[-1])
    return key_container, rank_container



def error_correction(prediction_table, all_keys):
    num_tasks = prediction_table.shape[0]
    key_container = copy.deepcopy(prediction_table)
    rank_container = np.zeros((num_tasks, num_tasks))
    # Error correction
    for a in range(num_tasks-1):
        for b in range(a+1, num_tasks):
            correct_key = all_keys[a]^all_keys[b]
            for c in range(num_tasks): 
                if c != a and c != b:
                    pred_container = np.zeros(256)
                    prediction_table[a, c] = min_max(prediction_table[a, c])
                    prediction_table[b, c] = min_max(prediction_table[b, c])   
                    
                    for k in range(256):                                                
                        pred = prediction_table[a, c] * prediction_table[b, c, np.bitwise_xor(range(256),k)]
                        pred_container[k] = np.max(pred)
                    key_container[a, b] *= pred_container

            rank_evol = 255-rk_key_eff([key_container[a, b]], correct_key)
            rank_container[a, b] = rank_evol[-1]
            rank_container[b, a] = rank_evol[-1]
    return rank_container


def perform_attacks(pred, plaintext, all_key, step=100, nb_attacks=1, shuffle=False):
    
    num_tasks = plaintext.shape[1]
    num_of_attack_traces = plaintext.shape[0]
    #pred_1 = np.log(pred_1+1e-40)
    #pred_2 = np.log(pred_2+1e-40)
    all_rk_evol = np.zeros((nb_attacks, num_tasks, int(num_of_attack_traces/step)))
    for idx in range(nb_attacks):
        print(idx)
        if shuffle:
            l = list(zip(pred, plaintext))
            random.shuffle(l)
            spred, splt = list(zip(*l))
            pred = np.array(spred)
            plaintext = np.array(splt, dtype=np.uint8)
        for byte in tqdm(range(num_tasks-1)):
            correct_key = all_key[byte]^all_key[byte+1]
            print('Correct key: ', correct_key)
            key_container = np.zeros((int(num_of_attack_traces/step), 256))
            # Perform attack
            for i in range(int(num_of_attack_traces/step)):
                # calculate averaged prediction based on each plaintext
                grouped_pred_1 = group_traces(pred[byte, :i*step], plaintext[:i*step, byte])
                grouped_pred_2 = group_traces(pred[byte+1, :i*step], plaintext[:i*step, byte+1])
                grouped_pred_1 = ss.rankdata(grouped_pred_1, method='dense', axis=-1) - 1
                grouped_pred_2 = ss.rankdata(grouped_pred_2, method='dense', axis=-1) - 1  
                for k in range(256):
                    conv_pred_1 = grouped_pred_1[np.transpose([range(grouped_pred_1.shape[0])]), np.bitwise_xor([range(256)],np.full((grouped_pred_1.shape[0], 1), k))]   
                    corr_list = corr2_coeff_rowwise2(conv_pred_1, grouped_pred_2[np.bitwise_xor(range(256),k)])
                    key_container[i, k] = np.mean(corr_list)

            key_container = np.cumsum(key_container, axis=0)
            print(f'best key: {np.argmax(key_container[-1])}')
            print(f'worst key: {np.argmin(key_container[-1])}')
            # Compute and plot the rank of the correct delta
            rank_evol = 255-rk_key_eff(key_container, correct_key)
            all_rk_evol[idx, byte] = rank_evol
            print("Final rank for differential attack: ", rank_evol[-1])
            print('GE smaller than 1:', np.argmax(rank_evol < 1))
            print('GE smaller than 5:', np.argmax(rank_evol < 5))

    return np.mean(all_rk_evol, axis=0)