import argparse
import os
import datetime
import time
import random

from rat_run import main
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--minutes", type=int, help="minutes of each trajectory")
parser.add_argument('-n', "--number", type=int, help="number of trajectories")
parser.add_argument('-p', "--parallel", type=bool, help='set True to use parallelized function')
args = parser.parse_args()

MINUTES = args.minutes
NUM_OF_RUNS = args.number
PARALLEL = args.parallel

PATH_DICT = {5: r'/home/mgteus/workspace/neuro/runs/5min/',
             10: r'/home/mgteus/workspace/neuro/runs/10min/',
             15: r'/home/mgteus/workspace/neuro/runs/15min/', }


def save_dfs_as_parquet(files_list: list, minutes: int = 5):
    init_writing_time = time.time()
    file_count = 0
    for df in files_list:
        # datetime format to YYYY-MM-DD_H_m_s
        file_tag = str(datetime.datetime.now())[:-4]
        file_tag = file_tag.replace(' ', '_')
        file_tag = file_tag.replace(':', '_')
        file_tag = file_tag.replace('.', '_')
        file_tag = file_tag + f'_r{random.randint(1,10000):05}'
        # filepath
        filename = file_tag + '_run_' + str(minutes)
        root_path = PATH_DICT[minutes]
        full_path = os.path.join(root_path, filename) + '.gzip'
        # saving dataframe
        df.to_parquet(full_path)
        file_count = file_count + 1
    print(f'  total write time of {file_count} files ', f'{time.time() - init_writing_time:3f}s')
    return


def run_main_parallel(num_of_runs, minutes):
    init_time = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(main)(minutes) for _ in range(num_of_runs)
    )
    print(f'  {len(results)}/{num_of_runs} runs completed')
    print(f'  runtime of {num_of_runs} {minutes}min runs = {time.time() - init_time:3f}s')

    save_dfs_as_parquet(results, minutes)

    return None


def run_main(num_of_runs: int = 100, minutes: int = 5):

    func_time = time.time()
    aux_dict = {i: None for i in range(num_of_runs)}

    for i in range(num_of_runs):
        init_time = time.time()
        aux_dict[i] = main(minutes)

        print(f'{i+1:2d} of {num_of_runs} {minutes}min runs',
              f'ts: {time.time() - init_time:3f}s')
    print('  total script time ', f'{time.time() - func_time:3f}s')

    print(f'saving {num_of_runs} files')
    savinf_file_init_time = time.time()
    for id_, df in aux_dict.items():
        file_tag = str(datetime.datetime.now())[:-4]
        file_tag = file_tag.replace(' ', '_').replace(':', '_').replace('.', '_')
        filename = file_tag + '_run_' + str(minutes)
        root_path = PATH_DICT[minutes]
        full_path = os.path.join(root_path, filename) + '.gzip'
        df.to_parquet(full_path)
    print('  total write time ', f'{time.time() - savinf_file_init_time:3f}s')
    return


if __name__ == '__main__':
    print(f'starting {"parallel" if PARALLEL else ""} simmulation of {NUM_OF_RUNS} {MINUTES}min runs [{datetime.datetime.now()}]')
    if PARALLEL:
        run_main_parallel(NUM_OF_RUNS, MINUTES)
    else:
        print('Running Serialized Code...')
        run_main(NUM_OF_RUNS, MINUTES)
    print(f'finished {"parallel" if PARALLEL else ""} simmulation of {NUM_OF_RUNS} {MINUTES}min runs [{datetime.datetime.now()}]')