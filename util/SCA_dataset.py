import numpy as np
import h5py

def load_ascad(ascad_database_file, profiling_traces=50000):
    in_file = h5py.File(ascad_database_file + 'ASCAD_nopoi_window_80.h5', "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.uint8)
    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'], dtype=np.uint8)
    K_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'], dtype=np.uint8)
    
    # Load attack traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    # Y_attack = np.array(in_file['Attack_traces/labels'], dtype=np.uint8)
    P_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'], dtype=np.uint8)
    K_attack = np.array(in_file['Attack_traces/metadata'][:]['key'], dtype=np.uint8)

    print('Profiling traces number: {}'.format(len(X_profiling)))
    print('Attack traces number: {}'.format(len(X_attack)))
    # return (X_profiling, X_attack), (Y_profiling,  Y_attack), (P_profiling,  P_attack)
    X_all = np.concatenate((X_profiling, X_attack), axis=0)
    # Y_all = np.concatenate((Y_profiling, Y_attack), axis=0)
    plt_all = np.concatenate((P_profiling, P_attack), axis=0)
    k_all = np.concatenate((K_profiling, K_attack), axis=0)
    print('Profiling traces number: {}'.format(len(X_all)))
    return X_all, plt_all

def load_ascad_rand(ascad_database_file, profiling_traces=60000):
    in_file = h5py.File(ascad_database_file + 'ascad-variable_nopoi_window_80.h5', "r")
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Y_profiling = np.array(in_file['Profiling_traces/labels'], dtype=np.uint8)
    P_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'], dtype=np.uint8)
    K_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'], dtype=np.uint8)
    
    key_init = np.array([0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255], dtype=np.uint8)
    simulated_key = np.repeat(key_init[np.newaxis, :], P_profiling.shape[0], axis=0)
    sim_plaintext = P_profiling ^ K_profiling ^ simulated_key
    return X_profiling[:profiling_traces], sim_plaintext[:profiling_traces]

def load_chesctf(database_file, target_byte=0, profiling_traces=50000):
    in_file = h5py.File(database_file+'ches_ctf_nopoi_window_80.h5', "r")
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
    plt_profiling = np.array(in_file['Profiling_traces/metadata'][:]['plaintext'], dtype=np.uint8)
    plt_attack = np.array(in_file['Attack_traces/metadata'][:]['plaintext'], dtype=np.uint8)
    k_profiling = np.array(in_file['Profiling_traces/metadata'][:]['key'], dtype=np.uint8)
    k_attack = np.array(in_file['Attack_traces/metadata'][:]['key'], dtype=np.uint8)
    print('Profiling traces number: {}'.format(np.shape(X_profiling)))
    print('Attack traces number: {}'.format(np.shape(X_attack)))
    
    X_all = np.concatenate((X_profiling, X_attack), axis=0)
    key_all = np.concatenate((k_profiling, k_attack), axis=0)
    plt_all = np.concatenate((plt_profiling, plt_attack), axis=0)
    print('All traces number: {}'.format(np.shape(X_all)))

    key_init = np.array([77, 251, 224, 242, 114, 33, 254, 16, 167, 141, 74, 220, 142, 73, 4, 105], dtype=np.uint8)
    simulated_key = np.repeat(key_init[np.newaxis, :], plt_all.shape[0], axis=0)
    sim_plaintext = plt_all ^ key_all ^ simulated_key
    return X_all, sim_plaintext