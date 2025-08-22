import torch
from multiprocessing import Process, Manager, Queue
from optimization.hyperparameter_tuning import tune_hyperparameters
from modeling.train_autoformer import train_autoformer
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown


def tune_etf0(etf, input_dir, output_dir, pred_len, target, n_trials, n_jobs, result_dict):
    """
    개별 ETF의 하이퍼파라미터 튜닝을 실행하는 함수
    """
    print(f"[▶] Starting hyperparameter tuning for {etf}")
    best_params = tune_hyperparameters(
        etf=etf,
        input_dir=input_dir,
        output_dir=output_dir,
        pred_len=pred_len,
        target=target,
        n_trials=n_trials,
        n_jobs=n_jobs
    )
    result_dict[etf] = best_params  # 결과 저장
    print(f"[✔] Best parameters for {etf}: {best_params}")

def tune_etf(etf, input_dir, output_dir, pred_len, target, n_trials, n_jobs, result_dict):
    """
    개별 ETF의 하이퍼파라미터 튜닝을 실행하는 함수
    """
    try:
        print(f"[▶] Starting hyperparameter tuning for {etf}")
        best_params = tune_hyperparameters(
            etf=etf,
            input_dir=input_dir,
            output_dir=output_dir,
            pred_len=pred_len,
            target=target,
            n_trials=n_trials,
            n_jobs=n_jobs
        )
        result_dict[etf] = best_params  # 결과 저장
        print(f"[✔] Best parameters for {etf}: {best_params}")
    except Exception as e:
        print(f"[⚠️] Error tuning hyperparameters for {etf}: {e}")

def parallel_tune(etf_list, input_dir, output_dir, pred_len, target, n_trials, n_jobs, gpu_ids):
    """
    병렬로 하이퍼파라미터 튜닝을 실행하는 함수
    """
    manager = Manager()
    result_dict = manager.dict()
    processes = []
    for i, etf in enumerate(etf_list):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # GPU ID 순환 할당
        wait_for_available_memory(gpu_id)  # GPU 메모리 확인 및 대기
        p = Process(target=tune_etf, args=(etf, input_dir, output_dir, pred_len, target, n_trials, n_jobs, result_dict))
        processes.append(p)
        p.start()

        # GPU 수에 맞게 작업 제한
        if len(processes) >= len(gpu_ids):
            for p in processes:
                p.join()
            processes = []

    # 남은 프로세스 종료 대기
    for p in processes:
        p.join()

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    print("[✔] All hyperparameter tuning processes completed.")
    return dict(result_dict)

def train_etf(etf, input_dir, output_dir, pred_len, target, best_params, gpu_id):
    """
    개별 ETF 학습을 위한 함수
    """
    wait_for_available_memory(gpu_id)
    print(f"[▶] Training {etf} with best hyperparameters on GPU {gpu_id}")
    train_autoformer(
        etf=etf,
        input_dir=input_dir,
        output_dir=output_dir,
        pred_len=pred_len,
        target=target,
        hyperparams=best_params
    )

def parallel_train(etf_list, input_dir, output_dir, pred_len, target, best_params_per_etf, gpu_ids):
    """
    병렬로 ETF 학습을 실행하는 함수

    Parameters:
    - etf_list: 학습할 ETF 리스트
    - input_dir: Autoformer 입력 데이터 경로
    - output_dir: Autoformer 출력 데이터 경로
    - pred_len: 예측 길이
    - target: 예측 대상 열 이름
    - best_params_per_etf: 각 ETF별 최적 하이퍼파라미터 딕셔너리
    - gpu_ids: 사용 가능한 GPU ID 리스트
    """
    processes = []
    for i, etf in enumerate(etf_list):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # GPU ID 순환 할당
        wait_for_available_memory(gpu_id)
        p = Process(target=train_etf, args=(etf, input_dir, output_dir, pred_len, target, best_params_per_etf[etf], gpu_id))
        processes.append(p)
        p.start()

        # GPU 수에 맞게 작업 제한
        if len(processes) >= len(gpu_ids):
            for p in processes:
                p.join()
            processes = []

    # 모든 프로세스가 종료될 때까지 대기
    for p in processes:
        p.join()
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    print("[✔] All training processes completed.")



def wait_for_available_memory(gpu_id, required_memory=4* 1024 * 1024 * 1024):  # 4GB 기본값
    """
    특정 GPU에서 사용 가능한 메모리가 충분할 때까지 대기
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_id)
    while True:
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem_info.free
        if free_memory >= required_memory:
            break
        print(f"[!] GPU {gpu_id} insufficient memory. Waiting...")
        time.sleep(5)  # 5초 대기
    nvmlShutdown()