import json
#from experiment.analysis import analyse_folder
#from experiment.source_dataset import adult
from tqdm import tqdm
import os
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import normalized_mutual_info_score

def binary_fmetrics_improvement(experiment, fair_column_mapping):
    values_sp = {}
    values_di = {}
    sd_with_nan = []
    
    for dataset in ['synth','train']:
        # Handle CSV reading based on whether file has header or not
        if experiment['parameters']['header'] is not None:
            # File has header, don't specify names, let pandas read from header
            exp_df = pd.read_csv(
                experiment['path'][dataset],
                sep=experiment['parameters']['sep'],
                header=experiment['parameters']['header'],
                usecols=experiment['parameters'].get('usecols'),
                dtype=experiment['parameters'].get('dtype', None)
            )
            # Reorder columns to match usecols order (CSV may have different column order)
            if 'usecols' in experiment['parameters']:
                exp_df = exp_df[experiment['parameters']['usecols']]
        else:
            # File has no header, must specify column names
            if 'usecols' in experiment['parameters']:
                names = experiment['parameters']['usecols']
            elif 'names' in experiment['parameters']:
                names = experiment['parameters']['names']
            else:
                raise ValueError("No column names found in experiment parameters.")
            exp_df = pd.read_csv(
                experiment['path'][dataset],
                names=names,
                sep=experiment['parameters']['sep'],
                header=None,
                dtype=experiment['parameters'].get('dtype', None)
            )
        target_column =  experiment['parameters']['target_column']
        # target_labels = [metric['label'] for metric in experiment['results']['tstr']['metrics']['synthetic']]
        target_labels = exp_df[target_column].dropna().unique().tolist()
        # Handle multiclass target by binarizing
        if len(target_labels) != 2:
            print(f"Warning: target variable has {len(target_labels)} unique values, expected 2. Binarizing target variable.")
            positive_class = experiment['parameters']['target_positiveclass']
            # Convert to list if it's a single string
            if isinstance(positive_class, str):
                positive_classes = [positive_class]
            else:
                positive_classes = positive_class
            # Create mapping: positive class(es) -> 1.0, all others -> 0.0
            binarization_mapping = {}
            for label in target_labels:
                if label.strip() in [pc.strip() for pc in positive_classes]:
                    binarization_mapping[label] = 1.0
                else:
                    binarization_mapping[label] = 0.0
            exp_df[target_column] = exp_df[target_column].map(binarization_mapping)
            print(f"  Mapped {positive_classes} to 1.0, all other classes to 0.0")
        else:
            # Binary target: map target labels to 0.0 and 1.0
            # Check that target_labels are in the expected order (1.0 must correspond to the positive class)
            positive_class = experiment['parameters']['target_positiveclass']
            # Handle both string and list format
            positive_class_str = positive_class[0] if isinstance(positive_class, list) else positive_class
            exp_df[target_column] = exp_df[target_column].map(
                {label: (1.0 if label.strip() == positive_class_str.strip() else 0.0) for label in target_labels}
            )
        for column_name, column_info in fair_column_mapping.items():
            if column_name in exp_df.columns:
                # Check for NaN values before mapping
                nan_mask_before = exp_df[column_name].isna()
                if nan_mask_before.any():
                    # nan_count = nan_mask_before.sum()
                    # print(f"Warning: Column '{column_name}' contains NaN values before mapping. Removing {nan_count} rows with NaN values.")
                    exp_df = exp_df[~nan_mask_before]   
                    # sd_with_nan.append((dataset, experiment['parameters']['time'], column_name, 'before_mapping', nan_count))
                
                ### UNCOMMENT THIS TO LOG UNMAPPED VALUES
                # Identify values not present in the mapping
                unique_values = exp_df[column_name].unique()
                mapping_keys = set(column_info['mapping'].keys())
                unmapped_values = [val for val in unique_values if val not in mapping_keys]  
                if unmapped_values:
                    print(f"Warning: Column '{column_name}' contains values not in mapping: {unmapped_values}")
                    print(f"  Available mapping keys: {list(mapping_keys)}")
                    sd_with_nan.append((dataset, experiment['parameters']['time'], column_name, 'unmapped_values', unmapped_values))
                
                exp_df[column_name] = exp_df[column_name].map(column_info['mapping'])
                
                # # Check for NaN values after mapping
                # nan_mask_after = exp_df[column_name].isna()
                # if nan_mask_after.any():
                #     nan_count = nan_mask_after.sum()
                #     print(f"Warning: Column '{column_name}' contains NaN values after mapping. Removing {nan_count} rows with NaN values.")
                #     exp_df = exp_df[~nan_mask_after]   
                #     sd_with_nan.append((dataset, experiment['parameters']['time'], column_name, 'after_mapping', nan_count))     
                
        columns_to_keep = [target_column] + list(fair_column_mapping.keys())
        exp_df = exp_df[columns_to_keep]

        bin_dataset = BinaryLabelDataset(
            favorable_label=1.0,
            unfavorable_label=0.0,
            df=exp_df,
            label_names=[target_column],
            # protected_attribute_names=experiment['parameters']['sensitive_features']
            protected_attribute_names=list(fair_column_mapping.keys())
        )
        # Estrai l'attributo sensibile (assumendo che ce ne sia uno solo)
        attr_name = list(fair_column_mapping.keys())[0]
        attr_info = fair_column_mapping[attr_name]
        privileged_groups = [{attr_name: val} for val in attr_info['privileged']]
        unprivileged_groups = [{attr_name: val} for val in attr_info['unprivileged']]
        metric = BinaryLabelDatasetMetric(bin_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        values_sp[dataset] = (metric.statistical_parity_difference())
        values_di[dataset] = (metric.disparate_impact())

    # Salva il contenuto di sd_with_nan in un file txt
    # if sd_with_nan:
    #     output_path = experiment['path']['datasets'].rsplit('/', 2)[0] + 'logs/nan_values_fair_report.txt'
    #     with open(output_path, 'w') as f:
    #         f.write("NaN Values Report\n")
    #         f.write("=" * 80 + "\n\n")
    #         f.write(f"Experiment time: {experiment['parameters']['time']}\n\n")
    #         f.write("Format: (dataset, experiment_time, column_name, nan_count)\n\n")
    #         for entry in sd_with_nan:
    #             f.write(f"{entry}\n")
    #     print(f"NaN values report saved to: {output_path}")

    return {"statistical_parity_difference": -(abs(values_sp['synth']) - abs(values_sp['train'])), "disparate_impact_difference": values_di['synth'] - values_di['train']}


def normalized_mutual_information(experiment, sensitive_feature):
    """
    Calculate Normalized Mutual Information (NMI) improvement between synthetic and training datasets.
    
    NMI measures statistical independence between a sensitive feature and the target variable:
    - NMI = 0: perfect independence (maximum fairness)
    - NMI = 1: complete dependence (no fairness)
    
    This metric does NOT require defining privileged/unprivileged groups and works with:
    - Binary or multiclass target variables
    - Binary or multiclass sensitive features
    
    Args:
        experiment (dict): Experiment configuration containing paths and parameters
        sensitive_feature (str): Name of the sensitive feature to analyze
        
    Returns:
        dict: Contains 'nmi_improvement' where:
              - Positive value: synthetic dataset is MORE fair (lower NMI)
              - Negative value: synthetic dataset is LESS fair (higher NMI)
    """
    nmi_values = {}
    
    for dataset in ['synth', 'train']:
        # Load dataset - handle CSV reading based on whether file has header or not
        if experiment['parameters']['header'] is not None:
            # File has header, don't specify names, let pandas read from header
            exp_df = pd.read_csv(
                experiment['path'][dataset],
                sep=experiment['parameters']['sep'],
                header=experiment['parameters']['header'],
                usecols=experiment['parameters'].get('usecols'),
                dtype=experiment['parameters'].get('dtype', None)
            )
            # Reorder columns to match usecols order (CSV may have different column order)
            if 'usecols' in experiment['parameters']:
                exp_df = exp_df[experiment['parameters']['usecols']]
        else:
            # File has no header, must specify column names
            if 'usecols' in experiment['parameters']:
                names = experiment['parameters']['usecols']
            elif 'names' in experiment['parameters']:
                names = experiment['parameters']['names']
            else:
                raise ValueError("No column names found in experiment parameters.")
            exp_df = pd.read_csv(
                experiment['path'][dataset],
                names=names,
                sep=experiment['parameters']['sep'],
                header=None,
                dtype=experiment['parameters'].get('dtype', None)
            )
        
        target_column = experiment['parameters']['target_column']
        
        # Check if sensitive feature exists
        if sensitive_feature not in exp_df.columns:
            raise ValueError(f"Sensitive feature '{sensitive_feature}' not found in dataset columns.")
        
        # Remove NaN values from both columns
        valid_mask = exp_df[sensitive_feature].notna() & exp_df[target_column].notna()
        exp_df_clean = exp_df[valid_mask]
        
        if len(exp_df_clean) == 0:
            raise ValueError(f"No valid rows after removing NaN values for '{sensitive_feature}' and '{target_column}'.")
        
        # Calculate NMI
        nmi = normalized_mutual_info_score(
            exp_df_clean[sensitive_feature],
            exp_df_clean[target_column],
            average_method='arithmetic'
        )
        
        nmi_values[dataset] = nmi
    
    # Return improvement: positive means synthetic is more fair (lower NMI)
    nmi_improvement = -(nmi_values['synth'] - nmi_values['train'])
    
    return {
        "nmi_improvement": nmi_improvement,
        "nmi_synthetic": nmi_values['synth'],
        "nmi_training": nmi_values['train']
    }


