#!/usr/bin/env python
# coding: utf-8

import os
import datetime
import logging
import logging.handlers
import pandas as pd
import multiprocessing
import queue
from experiment.experiment import Experiment
from experiment.source_dataset import *
from experiment.analysis import analyse_folder
import traceback
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str('false')
pd.set_option('display.max_columns', None)


def create_logger(log_folder):
    # now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    # log_folder = f"{folder}logs/{now}/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # create logger
    logger = logging.getLogger(f'engine')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.handlers.RotatingFileHandler(f"{log_folder}debug.log",
                                              maxBytes=2 * 1024 * 1024, backupCount=20)
    fh.namer = lambda name: name.replace(".log.", ".") + ".log"
    # self.fh = logging.FileHandler(f"{experiment_output_path}{experiment_code}.debug.log")
    fh.setLevel(logging.DEBUG)
    # create file handler which logs info messages
    fhi = logging.FileHandler(f"{log_folder}info.log")
    fhi.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fhi.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(fhi)
    logger.addHandler(ch)
    return logger

def create_experiment(loss_model, epochs_batch_size, learning_rate, decay_rate, 
                      simpson_weight, simpson_magnitude, simpson_exp_magnitude, 
                      only_categorical, layer_sizes, noy, split_random_state, 
                      folder_path, dataset_object, queue):
    try:
        time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        logger = create_logger(f'{folder_path}{time}/logs/')
        exp = Experiment(loss_model=loss_model, batch_size=epochs_batch_size[1], epochs=epochs_batch_size[0], learning_rate=learning_rate, decay=decay_rate, 
                         simpson_weight=simpson_weight, simpson_magnitude=simpson_magnitude, simpson_exp_magnitude=simpson_exp_magnitude, 
                         only_categorical=only_categorical, layers_size=layer_sizes, noy=noy, split_seed=split_random_state,
                         base_folder=folder_path, exp_id=time, source_dataset=dataset_object)
        exp.run()
    except Exception as e:
        queue.put(traceback.format_exc())
    return


if __name__ == "__main__":
    #NOTE Set to True to run the experiment, otherwise assign to run_experiment the folder path of the experiment to analyse
    #NOTE e.g. run_experiment = 'synthetic-datasets/01_adult/YYMMGG-HHMMSS-R-C/'
    # adult: 'synthetic-datasets/01_adult/250202-111013-30-22/'
    # south_german_credit: 'synthetic-datasets/03_south-german-credit/250202-111210-30-22/'
    # compas_score_two_years: 'synthetic-datasets/04_compas-two-years/260304-105454-30-14/'
    #NOTE Remember to adapt layer_size to the dataset
    run_experiment = True
    split_seeds = [0, 42 , 34192064, 75637324, 93015292, 134548530, 213281329, 222198395, 261053260, 330351956, 401612339, 411122582, 421541749, 428703832, 445596379, 563366835, 564253876, 566860962, 727205575, 727342120, 737841153, 787158527, 844424401, 860894124, 860982556, 866116827, 906595416, 915945376, 917004895, 963380211]
    repetitions = len(split_seeds)
    # adult, south_german_credit, compas_score_two_years
    datasets = [adult]
    # Adult 103 [80,50]       → hidden=78%, latent=49%
    # South German Credit 70 [48,32] → hidden=66%, latent=44%
    # Compas ~654 [467]   → Will be overwritten to [int(x_shape[0]/1.4)] due to high dimensionality
    layers_size = [[80,50]]
    learning_rates = [0.01]
    decays = [0.000001]
    loss_models = []
    loss_models.append(
        {'model': 'vanilla',
        'weights': [None],
        'magnitudes': [None],
        'exp_magnitudes': [None]}
    )
    loss_models.append(
        {'model': 'weight',
        'weights': [0.1,0.5,1.0], 
        'magnitudes': [1.0],
        'exp_magnitudes': [10.0]}
    )
    loss_models.append(
        {'model': 'term',
        'weights': [0.1,0.5,1.0],
        'magnitudes': [None],
        'exp_magnitudes': [None]}
    )
    only_cat = [False] # False for using all features, True for using only categorical features (if any)
    noys = [True] # True for considering the target column as a feature, False for not considering it as a feature
    
    experiment_base_path = "synthetic-datasets/"
    t = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    print(f'Testing {len(datasets)} datasets')
    folder_paths = []
    for experiment_dataset in datasets:
        print('* '* 20)
        print(f'Experiment for dataset {experiment_dataset["id"]}')
        print('* '* 20)
        epochs_and_batchsize = [(50,512), (1250, experiment_dataset["num_rows"])]
        loss_models_combinations = 0
        for lm in loss_models:
            loss_models_combinations += len(lm['weights']) * len(lm['magnitudes']) * len(lm['exp_magnitudes'])
        combinations = loss_models_combinations * len(epochs_and_batchsize) * len(learning_rates) * len(decays) * len(only_cat) * len(layers_size) * len(noys)
        eg_string = f'datasets: {experiment_dataset["id"]}, repetitions: {repetitions}, model: {loss_models}, epochs_and_batchsize: {epochs_and_batchsize}, learning_rates: {learning_rates}\n only_categorical = {only_cat}, layers_size = {layers_size}, noys = {noys}\n'
        experiment_dataset_name = experiment_dataset["id"]
        experiment_path = experiment_base_path + experiment_dataset_name + "/"  # synthetic-datasets/01_adult/
        folder_path = experiment_path + t + "-" + str(repetitions) + "-" + str(combinations) + "/"  # synthetic-datasets/01_adult/YYMMGG-HHMMSS-R-C/
        try:
            if run_experiment is True:
                for repetition in range(repetitions):
                    i = 0
                    split_seed = split_seeds[repetition]
                    print(f'Running repetition {repetition + 1} of {repetitions} [{split_seed}]')
                    # logger.info(f'Running repetition {repetition + 1} of {repetitions}')
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    with open(folder_path + 'experiment-group-info.txt', 'w') as file:
                        file.write(eg_string)
                    with open(folder_path + '_NOT-COMPLETED.txt', 'w') as file:
                        file.write('NOT COMPLETED')
                    for loss_model in loss_models:
                        for ebs in epochs_and_batchsize:
                            for lr in learning_rates:
                                for dc in decays:
                                    for oc in only_cat:
                                        for ls in layers_size:
                                            for noy in noys:
                                                for wt in loss_model['weights']:
                                                    for mt in loss_model['magnitudes']:
                                                        for xm in loss_model['exp_magnitudes']:
                                                            print('* ' * 20)
                                                            print(f'Running combination {i+1} of {combinations} (repetition {repetition + 1} of {repetitions} [{split_seed}])')    
                                                            queue = multiprocessing.Queue()
                                                            p = multiprocessing.Process(target=create_experiment, 
                                                                                        args=(loss_model['model'], ebs, lr, dc, 
                                                                                            wt, mt, xm, 
                                                                                            oc, ls, noy, split_seed,
                                                                                            folder_path, experiment_dataset, queue))
                                                            p.start()
                                                            p.join()
                                                            if not queue.empty():
                                                                raise Exception(queue.get())
                                                            i += 1
                
            else:
                folder_path = run_experiment
        except Exception as e:
            raise e
        else:
            experiments_df = analyse_folder(folder_path=folder_path, generate_plots=True, compare_exps=True)
            if os.path.exists(folder_path + '_NOT-COMPLETED.txt'):
                os.remove(folder_path + '_NOT-COMPLETED.txt')
            folder_paths.append(folder_path)
    if run_experiment is True and len(folder_paths) > 1:
        alldatasets_experiments_df = analyse_folder(folder_paths)



