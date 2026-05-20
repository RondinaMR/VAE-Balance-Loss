import os
from prettytable import PrettyTable
import logging
import pandas as pd
import json
from tqdm import tqdm
from experiment.plotting import plot_metrics, plot_latent_space, loss_plot_item
from experiment.utils import imbalance_statistics
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
logger = logging.getLogger(__name__)

def compare_imbalance(original, synthetic):
    my_table = PrettyTable()
    my_table.field_names = ["Feature", "D_original", "D_synthetic", "D_change", "m_original", "m_synthetic", "m_change"]
    for f in original['sensitive_features']:
        my_table.add_row([f, f"{original['results'][f]['simpson']:.2f}", f"{synthetic['results'][f]['simpson']:.2f}",
                          f"{synthetic['results'][f]['simpson'] - original['results'][f]['simpson']: .2f}",
                          len(original['frequencies'][f]), len(synthetic['frequencies'][f]),
                          len(synthetic['frequencies'][f]) - len(original['frequencies'][f])])
    return my_table.get_string()


def read_experiments(folder:str = None):
    if folder is None:
        folder = 'synthetic-datasets/01_adult/'
    directories = os.listdir(folder)

    experiments = []

    for exp in tqdm(directories, desc="Reading experiments"):
        if os.path.isdir(folder + exp) and exp != 'plots':
            try:
                file = open(f'{folder}{exp}/experiment.json')
            except FileNotFoundError:
                # print("ERROR: experiment.json not found in " + exp)
                logging.warning(f"experiment.json not found in {exp}: experiment not considered")
            else:
                with file:
                    experiment = json.load(file)
                    # check if 'path_to_json' is in the parameters
                    if 'path_to_json' not in experiment['parameters']:
                        experiment['parameters']['path_to_json'] = f'{folder}{exp}/experiment.json'
                    experiments.append(experiment['parameters'])
    df = pd.DataFrame.from_records(experiments)
    return df

# Imbalance_struct is a dictionary with the following structure:
# imbalance_struct = {
#     'results': {
#         'feature1': {
#             'gini': 0.5,
#             ...}}}


def summary_table(df, sort_by_pt):
    out_str = ""
    comparison_table = PrettyTable()
    comparison_table.field_names = ["dataset","t", "lm", "seed" , "noy", "bs", "e", "lr", "dc", "D_w", "D_m", "D_xm", "oc", "ls",
                                    "DCR S-H", "acc_s", "acc_diff", "D_train_loss", "D_synth_loss", "m_diff", "fit_time"]
    experiments = df.sort_values(by=["time"])['path_to_json'].tolist()
    for exp_item in experiments:
        with open(exp_item) as file:
            experiment = json.load(file)
            comparison_table.add_row([experiment['meta']['dataset_name'],
                                    experiment['parameters']['time'],
                                    experiment['parameters']['loss_model'],
                                    experiment['parameters']['split_seed'],
                                    experiment['parameters']['noy'],
                                    experiment['parameters']['batch_size'],
                                    experiment['parameters']['epochs'],
                                    experiment['parameters']['learning_rate'],
                                    experiment['parameters']['decay'],
                                    experiment['parameters']['simpson_weight'],
                                    experiment['parameters']['simpson_magnitude'],
                                    experiment['parameters']['simpson_exp_magnitude'],
                                    experiment['parameters']['only_categorical'],
                                    experiment['parameters']['layers_size'],
                                    f"{experiment['results']['synthetic_holdout_metrics']['synth_holdout_test']:.2f}",
                                    experiment['results']['tstr']['accuracy']['synthetic'],
                                    f"{experiment['results']['tstr']['accuracy']['synthetic'] - experiment['results']['tstr']['accuracy']['training']: .4f}",
                                    f"{imbalance_statistics(experiment['results']['training_imbalance'], stat='sensitiveloss', sensitive_features=experiment['parameters']['sensitive_features']):.2f}",
                                    f"{imbalance_statistics(experiment['results']['synthetic_imbalance'], stat='sensitiveloss', sensitive_features=experiment['parameters']['sensitive_features']):.2f}",
                                    f"{imbalance_statistics(experiment['results']['synthetic_imbalance'], stat='m_change') - imbalance_statistics(experiment['results']['training_imbalance'], stat='m_change'):.2f}",
                                    f"{experiment['meta']['fit_time']:.0f}"]
                                    )
    out_str += "* Summary of experiments\n"
    out_str += comparison_table.get_string(sortby=sort_by_pt) #, reversesort=True
    out_str += f"\n\n# of processed experiments: {len(experiments)}."
    return out_str

def mixed_stats(experiment):
    out_str = ""
    table = PrettyTable()
    table.field_names=["dataset", "duplicates", "unique_duplicates"]
    table.add_row([
        "training",
        experiment['results']['training_metrics']['training_duplicates'],
        experiment['results']['training_metrics']['training_unique_duplicates'],
    ])
    table.add_row([
        "synthetic",
        experiment['results']['synthetic_metrics']['synthetic_duplicates'],
        experiment['results']['synthetic_metrics']['synthetic_unique_duplicates'],
    ])
    out_str += table.get_string()
    out_str += f"\n\n# of clones synthetic-training: {experiment['results']['synthetic_training_metrics']['synth_train_clones']} ({experiment['results']['synthetic_training_metrics']['synth_train_clones_percentage']:.2f}%).\n"    
    return out_str

def compare_experiments(df, folder, sort_by_pt):
    with open(f'{folder}report.txt', 'w') as file_report:
       
        experiments = df.sort_values(by=["time"])['path_to_json'].tolist()
        # by=["time", "model_version", "batch_size", "epochs", "learning_rate", "simpson_weight"]
        for exp_item in experiments:
            with open(exp_item) as file:
                experiment = json.load(file)
                # if 'model_version' not in experiment['parameters']:
                #     experiment['parameters']['model_version'] = '?'
                file_report.write(f"* Experiment {experiment['parameters']}\n")
                file_report.write(f"\n**** Mixed stats\n")
                file_report.write(mixed_stats(experiment))
                # print(f"Training duplicates: {experiment['results']['training_metrics']['training_duplicates']}")
                # print(f"Synthetic duplicates: {experiment['results']['synthetic_metrics']['synthetic_duplicates']}")
                # print(f"Synthetic-Training clones: {experiment['results']['synthetic_training_metrics']['synth_train_clones']}")
                file_report.write(f"\n**** DCR\n")
                # print(f"Synthetic-Holdout test: {experiment['results']['synthetic_holdout_metrics']['synth_holdout_test']:.2f}")
                table = PrettyTable()
                table.field_names = ["Synthetic-Holdout test"]
                table.add_row([f"{experiment['results']['synthetic_holdout_metrics']['synth_holdout_test']:.2f}"])
                file_report.write(table.get_string())
                # print(f"Adversary precision: {experiment['results']['membership_inference_test']['adversary_precisions']}")
                # print(f"Membership Inference Mean Risk Score: {experiment['results']['membership_inference_test']['membership_inference_mean_risk_score']}")
                # print(f"Detection Score: {experiment['results']['detection_score']}")
                file_report.write(f"\n\n**** TSTR Score\n")
                file_report.write(f"\n\n**** **** Accuracy\n")
                table = PrettyTable()
                table.field_names = ["Accuracy_original", "Accuracy_synthetic", "change", "score"]
                table.add_row([experiment['results']['tstr']['accuracy']['training'],
                            experiment['results']['tstr']['accuracy']['synthetic'],
                            f"{experiment['results']['tstr']['accuracy']['synthetic'] - experiment['results']['tstr']['accuracy']['training']: .2f}",
                            f"{experiment['results']['tstr']['score']: .2f}"
                            ])
                file_report.write(table.get_string())

                file_report.write(f"\n\n**** **** Metrics\n")
                table = PrettyTable()
                table.field_names = ["Dataset", "Label", "Precision", "Recall", "F1-score", "Support"]
                for dataset in ['training', 'synthetic']:
                    for target_class in experiment['results']['tstr']['metrics'][dataset]:
                        table.add_row([dataset, 
                                    target_class['label'], 
                                    target_class['precision'], 
                                    target_class['recall'],
                                    target_class['fscore'],
                                    target_class['support']])
                file_report.write(table.get_string(sortby="Label"))
                # print(f"Original Mutual Information: {experiment['results']['mutual_information']['original_mutual_information']}")
                # print(f"Feature Comparison: {experiment['results']['feature_comparison']}")
                # print(f"Query Power: {experiment['results']['query_power']}")
                # print(f"Reconstruction Error: {experiment['results']['reconstruction_error']}")
                # print(f"Training Imbalance: {experiment['results']['training_imbalance']}")
                # print(f"Synthetic Imbalance: {experiment['results']['synthetic_imbalance']}")
                file_report.write(f"\n\n**** Imbalance\n")
                file_report.write(compare_imbalance(experiment['results']['training_imbalance'], experiment['results']['synthetic_imbalance']))
                file_report.write("\n\n")
        file_report.write(summary_table(df=df, sort_by_pt=sort_by_pt))
    with open(f'{folder}report.txt', 'r') as file_report:
        print(file_report.read())
    return


def plot_experiments(df):
    print('Plotting latent space...')
    experiments = df.sort_values(by=["time"])['path_to_json'].tolist()
    for exp_item in experiments:
        with open(exp_item) as file:
            experiment = json.load(file)
            plot_latent_space(experiment)
    return

def hypertune(df, tune_parameters):
    print(df.head())
    data = []
    for _, row in df.iterrows():
        with open(row['path_to_json']) as file:
            experiment = json.load(file)
            data.append({
                'time': row['time'],
                'batch_size': experiment['parameters']['batch_size'],
                'epochs': experiment['parameters']['epochs'],
                'learning_rate': experiment['parameters']['learning_rate'],
                'decay': experiment['parameters']['decay'],
                'layers_size': experiment['parameters']['layers_size'],
                'acc_diff': experiment['results']['tstr']['accuracy']['synthetic'] - experiment['results']['tstr']['accuracy']['training'],
                'DCR S-H': experiment['results']['synthetic_holdout_metrics']['synth_holdout_test'],
                'm_diff': imbalance_statistics(experiment['results']['synthetic_imbalance'], stat='m_change') - imbalance_statistics(experiment['results']['training_imbalance'], stat='m_change')
            })
    df_tuned = pd.DataFrame(data)
    pt_fields_to_tune = ['batch_size', 'epochs', 'learning_rate', 'decay', 'layers_size']
    target_variables = ['m_diff', 'acc_diff', 'DCR S-H']
    df_tuned['layers_size'] = df_tuned['layers_size'].apply(lambda x: f"[{x[0]},{x[1]}]" if isinstance(x, list) and len(x) == 2 else str(x))
    if 1 in tune_parameters:
        for field in pt_fields_to_tune:
            # for target in target_variables:
            avg_target = df_tuned.groupby(field)[target_variables].mean().reset_index().sort_values(by=['m_diff', 'acc_diff', 'DCR S-H'])
            print(f"Average {target_variables} for each {field}:")
            print(avg_target)
    if 2 in tune_parameters:
        for i in range(len(pt_fields_to_tune)):
            for j in range(i + 1, len(pt_fields_to_tune)):
                field1 = pt_fields_to_tune[i]
                field2 = pt_fields_to_tune[j]
                for target in target_variables:
                    avg_target = df_tuned.groupby([field1, field2])[target].mean().reset_index().sort_values(by=target)
                    print(f"Average {target} for each combination of {field1} and {field2}:")
                    print(avg_target)
    if 3 in tune_parameters:
        for i in range(len(pt_fields_to_tune)):
            for j in range(i + 1, len(pt_fields_to_tune)):
                for k in range(j + 1, len(pt_fields_to_tune)):
                    field1 = pt_fields_to_tune[i]
                    field2 = pt_fields_to_tune[j]
                    field3 = pt_fields_to_tune[k]
                    for target in target_variables:
                        avg_target = df_tuned.groupby([field1, field2, field3])[target].mean().reset_index().sort_values(by=target)
                        print(f"Average {target} for each combination of {field1}, {field2}, and {field3}:")
                        print(avg_target)
    if 4 in tune_parameters:
        for i in range(len(pt_fields_to_tune)):
            for j in range(i + 1, len(pt_fields_to_tune)):
                for k in range(j + 1, len(pt_fields_to_tune)):
                    for l in range(k + 1, len(pt_fields_to_tune)):
                        field1 = pt_fields_to_tune[i]
                        field2 = pt_fields_to_tune[j]
                        field3 = pt_fields_to_tune[k]
                        field4 = pt_fields_to_tune[l]
                        # for target in target_variables:
                        avg_target = df_tuned.groupby([field1, field2, field3, field4])[target_variables].mean().reset_index().sort_values(by=['m_diff', 'acc_diff', 'DCR S-H'])
                        print(f"Average for each combination of {field1}, {field2}, {field3}, and {field4}:")
                        print(avg_target)

    return


def analyse_folder(folder_path, sort_by_pt="t", generate_plots=False, compare_exps=False, tune_parameters=False):
    logger.info('Analyzing folder: %s', folder_path)
    print(f"Analyzing folder: {folder_path}")
    """
    Analyzes the experiments in the specified folder and compares them.

    Parameters:
    folder_path (str, list): The path to the folder containing the experiment data or a list of paths.
    sort_by_pt (str, optional): The parameter to sort the experiments by (PrettyTable field name). Defaults to "t".
    sensitive_features (list, optional): A list of sensitive features to consider during analysis. Defaults to None.

    Returns:
    DataFrame: A DataFrame containing the experiment data.
    """
    if type(folder_path) is str:
        experiments_df = read_experiments(folder=folder_path)
        # experiments_df = experiments_df[experiments_df['only_categorical'] == False]
        if generate_plots:
            plot_experiments(experiments_df)
            loss_plot_item(experiments_df)
            plot_metrics(experiments_df, folder_path)
        if compare_exps != False:
            compare_experiments(experiments_df, folder=folder_path, sort_by_pt=sort_by_pt)
        if tune_parameters != False:
            hypertune(experiments_df, tune_parameters=tune_parameters)
    elif type(folder_path) is list:
        experiments_df = pd.DataFrame()
        for folder in folder_path:
            experiments_df = pd.concat([experiments_df, read_experiments(folder=folder)], ignore_index=True)
            if generate_plots:
                plot_experiments(experiments_df, folder=folder)
                loss_plot_item(experiments_df)
                plot_metrics(experiments_df, folder_path)
        print(summary_table(experiments_df, sort_by_pt))
        if tune_parameters != False:
            hypertune(experiments_df, tune_parameters=tune_parameters)
    else:
        raise ValueError("folder_path must be a string or a list of strings.")
    return experiments_df



# if __name__ == "__main__":
#     #NOTE 'synthetic-datasets/01_adult/240909-170231-1/' 864 experiments of independent error loss (sum)
#     folder_path = 'synthetic-datasets/03_south-german-credit/250202-111210-30-22/'
#     experiments_df = analyse_folder(folder_path, sort_by_pt="t") #sort_by_pt="acc_diff", tune_parameters=[1,4]
#     plot_metrics(experiments_df, folder_path)