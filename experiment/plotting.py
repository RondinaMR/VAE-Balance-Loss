import json
import pandas as pd
import matplotlib.pyplot as plt
from experiment.utils import imbalance_statistics
from experiment.fairness import binary_fmetrics_improvement
from experiment.fairness import normalized_mutual_information
import os
import math
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm

SPD_YLIM = {
    '01_adult': (-0.15, 0.25),
    '03_south-german-credit': (-0.10, 0.25),
    '04_compas-two-years': (-0.36, 0.30),
}


def separate_pathtojson_by_simpsonweight(df):
    df_wd0 = df[df['simpson_weight'] == 0]
    df_wd1 = df[df['simpson_weight'] == 1]
    path_to_json_list_wd0 = df_wd0['path_to_json'].tolist()
    path_to_json_list_wd1 = df_wd1['path_to_json'].tolist()
    return path_to_json_list_wd0, path_to_json_list_wd1

def separate_feature_simpson_by_simpsonweight(df, folder, feature):
    path_to_json_list_wd0, path_to_json_list_wd1 = separate_pathtojson_by_simpsonweight(df)
    feature_simpsons = {}
    feature_simpsons['wd0'] = []
    feature_simpsons['wd1'] = []

    for group_type in ['wd0', 'wd1']:
        if group_type == 'wd0':
            group_type_path = path_to_json_list_wd0
        else:
            group_type_path = path_to_json_list_wd1
        for exp_item in group_type_path:
            with open(exp_item) as file:
                experiment = json.load(file)
                feature_simpsons[group_type].append(experiment['results']['synthetic_imbalance']['results'][feature]['simpson'])
    return feature_simpsons['wd0'], feature_simpsons['wd1']

def boxplot_wd0_wd1(df, folder, feature):
    with open(f'{folder}experiment-group-info.txt', 'r') as file:
        experiment_description_str = file.read()
    feature_simpsons = {}
    feature_simpsons['wd0'] = []
    feature_simpsons['wd1'] = []
    feature_simpsons['wd0'], feature_simpsons['wd1'] = separate_feature_simpson_by_simpsonweight(df, folder, feature)
    fig, ax = plt.subplots()
    ax.boxplot([feature_simpsons['wd0'], feature_simpsons['wd1']])
    ax.set_xticklabels(['wd0', 'wd1'])
    ax.set_ylabel('Simpson Index')
    ax.set_title(f'Comparison of Simpson Index between wd0 and wd1 ({feature})\n{experiment_description_str}')    
    plt.savefig(f'{folder}_boxplot_{feature}_wd0-wd1.pdf')
    plt.close()

def scatter_xwd0_ywd1(df, folder, feature):
    with open(f'{folder}experiment-group-info.txt', 'r') as file:
        experiment_description_str = file.read()
    feature_simpsons = {}
    feature_simpsons['wd0'] = []
    feature_simpsons['wd1'] = []
    feature_simpsons['wd0'], feature_simpsons['wd1'] = separate_feature_simpson_by_simpsonweight(df, folder, feature)
    fig, ax = plt.subplots()
    ax.scatter(feature_simpsons['wd0'], feature_simpsons['wd1'])
    ax.set_xlabel('Simpson Index wd0')
    ax.set_ylabel('Simpson Index wd1')
    ax.set_title(f'Scatter plot of Simpson Index between wd0 and wd1 ({feature})\n{experiment_description_str}')
    plt.savefig(f'{folder}_scatter_{feature}_wd0-wd1.pdf')
    plt.close()

def loss_plot_item(df):
    logger.info('Plotting loss for individual experiments...')
    experiments_df = df
    path_to_json_list = experiments_df['path_to_json'].tolist()
    for exp_item in tqdm(path_to_json_list, desc="Plotting loss for individual experiments"):
        item_path = exp_item.replace('experiment.json', '')
        losses_df = pd.read_csv(f'{item_path}epoch-losses.csv')
        plt.plot(losses_df['this_epoch'].to_list(), losses_df['loss'].to_list(), label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.savefig(f'{item_path}epoch_losses.pdf')
        plt.close()
    return

def loss_plot_experiments(df, folder):
    path_to_json_list_wd0, path_to_json_list_wd1 = separate_pathtojson_by_simpsonweight(df)
    plt.figure()
    for path_list in [path_to_json_list_wd0, path_to_json_list_wd1]:
        if path_list == path_to_json_list_wd0:
            wd = 0
        else:
            wd = 1
        for exp_item in path_list:
            item_path = exp_item.replace('experiment.json', '')
            losses_df = pd.read_csv(f'{item_path}epoch-losses.csv')
            plt.plot(losses_df['this_epoch'].to_list(), losses_df['loss'].to_list(), label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.savefig(f'{folder}epoch_losses_wd{wd}.pdf')
        plt.close()
    return



def drop_non_finite(values, labels, metric, epochs):
    """Drop NaN and infinite values from each boxplot series.

    matplotlib computes the percentiles over the whole series, so a single NaN makes the entire box disappear without raising anything. 
    This happens for instance when the synthetic dataset of one seed contains no row of the privileged group, which leaves the statistical parity difference undefined, or when the disparate impact divides by zero.
    Discarding those points keeps the box, drawn over the remaining seeds, and the number of discarded ones is reported so it can be stated alongside the figure.
    """
    cleaned = []
    for label, series in zip(labels, values):
        finite = [value for value in series if value is not None and math.isfinite(value)]
        dropped = len(series) - len(finite)
        if dropped:
            message = (f'{metric} (epochs={epochs}), {label}: {dropped} of {len(series)} values are '
                       f'NaN or infinite and are excluded, the box is drawn over {len(finite)} seeds')
            logger.warning(message)
            print(f'  [plot_metrics] {message}')
        cleaned.append(finite)
    return cleaned


def plot_metrics(df, folder):
    logger.info('Plotting metrics...')
    print('Plotting metrics...')
    for epochs in tqdm([50, 1250], desc="Plotting metrics for different epochs"):
        edf = df[df['epochs'] == epochs]
        epochs_string = f'e{epochs}'
        if epochs == 50:
            batch_string = '512'
        else:
            batch_string = 'monobatch'
        metrics = {
            'synth_holdout_test': [],
            'tstr_accuracy_diff': [],
            'imbalance_loss_diff': [],
            'imbalance_loss_final': []
        }
        
        # Retrieve fair_column_mappings from the first experiment
        if edf.empty:
            logger.warning(f'No experiments found for epochs={epochs}')
            continue
        
        first_experiment_path = edf['path_to_json'].iloc[0]
        with open(first_experiment_path) as file:
            first_exp = json.load(file)
            fair_column_mappings = first_exp['parameters']['fair_column_mappings']
            # imbalance_loss_final sums (1 - Simpson) over the sensitive features, so the number of those features is the upper bound of its scale: 3 for adult and compas, 6 for south german credit.
            n_sensitive_features = len(first_exp['parameters']['sensitive_features'])
        
        # Verify that all experiments have the same fair_column_mappings
        for exp_path in edf['path_to_json']:
            with open(exp_path) as file:
                exp = json.load(file)
                if exp['parameters']['fair_column_mappings'] != fair_column_mappings:
                    raise ValueError(f"Inconsistent fair_column_mappings detected! "
                                   f"Expected {fair_column_mappings} but found "
                                   f"{exp['parameters']['fair_column_mappings']} in {exp_path}")

        # Add a key for each attribute in fair_column_mappings to store the statistical parity difference improvement values
        for attr_name in fair_column_mappings.keys():
            metrics[f'statistical_parity_diff_improvement_{attr_name}'] = []
            metrics[f'disparate_impact_diff_{attr_name}'] = []
            # metrics[f'nmi_improvement_{attr_name}'] = []
        
        labels = []

        for loss_model in tqdm(['vanilla', 'weight', 'term'], desc=f"Processing models (epochs={epochs})", leave=False):
            for simpson_weight in tqdm([None, 0.1, 0.5, 1.0], desc=f"Processing simpson weights for model {loss_model}", leave=False):
                if loss_model == 'vanilla' and simpson_weight == None:
                    # if simpson_weight != None:
                    #     continue
                    subset_df = edf[edf['loss_model'] == loss_model]
                else:
                    subset_df = edf[(edf['loss_model'] == loss_model) & (edf['simpson_weight'] == simpson_weight)]
                if subset_df.empty:
                    continue
                experiments = subset_df['path_to_json'].tolist()
                synth_holdout_test = []
                tstr_accuracy_diff = []
                imbalance_loss_diff = []
                imbalance_loss_final = []
                # Create a dictionary for each attribute
                statistical_parity_diff = {attr_name: [] for attr_name in fair_column_mappings.keys()}
                disparate_impact_diff = {attr_name: [] for attr_name in fair_column_mappings.keys()}
                # nmi_improvements = {attr_name: [] for attr_name in fair_column_mappings.keys()}
                for exp_item in tqdm(experiments, desc=f"Processing experiments", leave=False):
                    with open(exp_item) as file:
                        experiment = json.load(file)
                        synth_holdout_test.append(experiment['results']['synthetic_holdout_metrics']['synth_holdout_test'])
                        tstr_accuracy_diff.append(experiment['results']['tstr']['accuracy']['synthetic'] - experiment['results']['tstr']['accuracy']['training'])
                        imbalance_loss_diff.append(abs(imbalance_statistics(experiment['results']['synthetic_imbalance'], stat='sensitiveloss', sensitive_features=experiment['parameters']['sensitive_features']) - imbalance_statistics(experiment['results']['training_imbalance'], stat='sensitiveloss', sensitive_features=experiment['parameters']['sensitive_features'])))
                        imbalance_loss_final.append(imbalance_statistics(experiment['results']['synthetic_imbalance'], stat='sensitiveloss', sensitive_features=experiment['parameters']['sensitive_features']))
                        
                        # Calculate statistical_parity_difference for each attribute
                        for attr_name, attr_info in fair_column_mappings.items():
                            fair_column_mapping = {attr_name: attr_info}
                            statistical_parity_diff[attr_name].append(binary_fmetrics_improvement(experiment, fair_column_mapping=fair_column_mapping)["statistical_parity_difference"])
                            disparate_impact_diff[attr_name].append(binary_fmetrics_improvement(experiment, fair_column_mapping=fair_column_mapping)["disparate_impact_difference"])
                            # nmi_improvements[attr_name].append(normalized_mutual_information(experiment, attr_name)['nmi_improvement'])
                
                metrics['synth_holdout_test'].append(synth_holdout_test)
                metrics['tstr_accuracy_diff'].append(tstr_accuracy_diff)
                metrics['imbalance_loss_diff'].append(imbalance_loss_diff)
                metrics['imbalance_loss_final'].append(imbalance_loss_final)
                for attr_name in fair_column_mappings.keys():
                    metrics[f'statistical_parity_diff_improvement_{attr_name}'].append(statistical_parity_diff[attr_name])
                    metrics[f'disparate_impact_diff_{attr_name}'].append(disparate_impact_diff[attr_name])
                    # metrics[f'nmi_improvement_{attr_name}'].append(nmi_improvements[attr_name])
                if loss_model == 'vanilla':
                    labels.append(f'vanilla')
                else:
                    labels.append(f'{loss_model}_{simpson_weight}')

        for metric, values in metrics.items():
            fig, ax = plt.subplots(figsize=(12, 8))  # Increase the horizontal width
            ax.boxplot(drop_non_finite(values, labels, metric, epochs), labels=labels)
            ax.set_ylabel(metric.replace('_', ' ').title().replace('Statistical Parity Diff', 'S.P.D.'), fontsize=26)
            ax.set_xlabel('Model and Magnitude', fontsize=26)
            # ax.set_title(f'{metric.replace("_", " ").title()} for Different Models and Magnitude (epochs: {epochs}; batch size: {batch_string})', fontsize=16)
            ax.tick_params(axis='x', labelsize=22)  # Increment the font size of the x ticks
            plt.xticks(rotation=15)  # Rotate the x tick labels by 15 degrees
            ax.tick_params(axis='y', labelsize=22)  # Increment the font size of the y ticks
            # Set y-axis limits for statistical parity difference plots
            if 'statistical_parity' in metric:
                # Per-dataset scale: within a dataset it must stay identical across attributes and epoch settings, which is the comparison the figures are read for.
                # Compas needs a wider window since its positive class is Low only (observed [-0.350, +0.296]) than adult [-0.142, +0.204] and german [-0.063, +0.239].
                ylim = next((v for k, v in SPD_YLIM.items() if k in folder), None)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1)
            elif 'imbalance_loss_final' in metric:
                # The scale is the metric range, not a display choice, so it follows the dataset instead of being fixed at 3.
                ax.set_ylim(-0.1, n_sensitive_features + 0.1)
            elif 'tstr_accuracy_diff' in metric:
                ax.set_ylim(-0.21, 0.11)
                ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1)
            elif 'synth_holdout_test' in metric:
                ax.set_ylim(34, 71)
                ax.axhline(y=50, color='red', linestyle='--', linewidth=1)
            ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1)
            ax.axvline(x=4.5, color='gray', linestyle='--', linewidth=1)
            fig.tight_layout(pad=0.1)  # Reduce padding
            if not os.path.exists(f'{folder}plots/'):
                os.makedirs(f'{folder}plots/')
            plt.savefig(f'{folder}plots/boxplot_{epochs_string}_{metric}.pdf')
            plt.close(fig)

    return

def plot_latent_space(experiment):
#         b2 = np.load(exper)
#         # Create a scatter plot with b2
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(10, 8))
#         plt.scatter(b2[:, 0], b2[:, 1], alpha=0.5)
#         plt.title('Latent Space Distribution')
#         plt.xlabel('Latent Dimension 1')
#         plt.ylabel('Latent Dimension 2')
#         plt.savefig(f'{self.experiment_output_path_viz}latent_space.pdf')
#         plt.close()

#         # Create a heatmap with b2
#         plt.figure(figsize=(12, 10))
#         plt.imshow(b2, cmap='hot', aspect='auto')
#         plt.title('Heatmap of Latent Space Distribution')
#         plt.xlabel('Latent Dimensions')
#         plt.ylabel('Samples')
#         plt.savefig(f'{self.experiment_output_path_viz}latent_space_heatmap.pdf')
#         plt.close()

#         # Create a histogram of frequency of values in b2
#         plt.figure(figsize=(10, 8))
#         plt.hist(b2.ravel(), bins=50, alpha=0.75, color='blue', edgecolor='black')
#         plt.title('Histogram of Latent Space Values')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
#         plt.savefig(f'{self.experiment_output_path_viz}latent_space_histogram.pdf')
#         plt.close()
    return

def plot_x_imbalance_y_accuracy():
    return
