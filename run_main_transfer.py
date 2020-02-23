import subprocess
from multiprocessing import Pool
import os


def train(dataset, save_dir, result_dir, save_z_flag=0, get_score_flag=1, restore_dir=None, max_epoch=10,
          GPU_device_number='-1', untrainable_variables_keyvalues=None):
    args = ['python', 'main_transfer.py', '--dataset', dataset, '--save_dir', save_dir, '--result_dir', result_dir]
    if save_z_flag == 1:
        args.extend(['--save_z', '1'])
    if get_score_flag == 0:
        args.extend(['--get_score_for_each_machine_flag', '0'])
    if untrainable_variables_keyvalues is not None:
        args.extend(['--untrainable_variables_keyvalues', untrainable_variables_keyvalues])
    if GPU_device_number != '-1':
        args.extend(['--GPU_device_number', GPU_device_number])
    if restore_dir is not None:
        args.extend(['--restore_dir', restore_dir])
    if max_epoch != 10:
        args.extend(['--max_epoch', str(max_epoch)])
    print(f'running program {args}.')
    subprocess.call(args)


def run_train(args):
    train(*args)


def train_part1():
    machine_list = [f'machine-1-{k}' for k in range(1, 9)] + [f'machine-2-{k}' for k in range(1, 10)] + \
                   [f'machine-3-{k}' for k in range(1, 12)]
    dataset = ','.join(machine_list)
    train(dataset, 'model0', 'result_for_period1', get_score_flag=0, GPU_device_number='1,2,3')


def get_all_machines_z():
    save_z_flag, get_score_flag, restore_dir, max_epoch = 1, 1, 'model0', 0
    machine_list = [f'machine-1-{k}' for k in range(1, 9)] + [f'machine-2-{k}' for k in range(1, 10)] + \
                   [f'machine-3-{k}' for k in range(1, 12)]
    all_GPU_number = 8
    pool = Pool(all_GPU_number)
    for i in range(int(len(machine_list) / all_GPU_number) + 1):
        machine_list_part = machine_list[i*all_GPU_number:(i+1)*all_GPU_number]
        args_list = [
            (m, restore_dir, 'result_for_period1', save_z_flag, get_score_flag, restore_dir, max_epoch, str(GPU_number))
            for GPU_number, m in enumerate(machine_list_part)
        ]
        pool.map(run_train, args_list)


def train_part2():
    all_machine_list = [f'machine-1-{k}' for k in range(1, 9)] + [f'machine-2-{k}' for k in range(1, 10)] + \
                       [f'machine-3-{k}' for k in range(1, 12)]
    machine_list = [
        [all_machine_list[j] for j in [0, 5, 6, 7, 10, 11, 13, 17, 18, 19, 21, 22, 23, 25]],
        [all_machine_list[j] for j in [2, 3, 4, 8, 9, 12, 14, 15, 16, 20, 24]],
    ]
    print(machine_list)
    args_list = [
        (','.join(m), f'model{i+1}', f'result_for_period2_part{i+1}', 0, 1, None, 10, str(i), "rnn_q_z")
        for i, m in enumerate(machine_list)
    ]
    pool = Pool(len(machine_list))
    pool.map(run_train, args_list)


train_part1()
get_all_machines_z()
train_part2()