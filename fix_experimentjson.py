import json
from experiment.analysis import analyse_folder
from experiment.source_dataset import adult,south_german_credit,compas_score_two_years
from tqdm import tqdm

if __name__ == "__main__":
    # adult: 'synthetic-datasets/01_adult/250202-111013-30-22/'
    # south_german_credit: 'synthetic-datasets/03_south-german-credit/250202-111210-30-22/'
    # compas_score_two_years: 'synthetic-datasets/04_compas-two-years/250226-180048-30-22/'
    # bank_marketing: 'synthetic-datasets/02_bank-marketing/250202-111126-30-22/'
    # compas_score_two_years (run used by statistical_validation.py): 'synthetic-datasets/04_compas-two-years/260304-105454-30-14/'
    folder = 'synthetic-datasets/01_adult/250202-111013-30-22/'
    source_dataset = adult
    experiments_df = analyse_folder(folder_path=folder)
    experiments = experiments_df.sort_values(by=["time"])['path_to_json'].tolist()
    print(f'Fixing {len(experiments)} experiments')
    for exp_item in tqdm(experiments, desc="Fixing experiments"):
        with open(exp_item) as file:
            experiment = json.load(file)
            # Add usecols to parameters
            if 'usecols' in source_dataset:
                experiment['parameters']['usecols'] = source_dataset['usecols']
            # add names to parameters
            experiment['parameters']['names'] = source_dataset['names']
            # add sep to parameters
            experiment['parameters']['sep'] = source_dataset['sep']
            # add header to parameters
            experiment['parameters']['header'] = source_dataset['header']
            # add dtype to parameters
            if 'dtype' in source_dataset:
                experiment['parameters']['dtype'] = source_dataset['dtype']
            # Add target_column to parameters
            experiment['parameters']['target_column'] = source_dataset['target_column']
            # experiment['meta']['target_column'] = source_dataset['target_column']
            # if 'meta' in experiment and 'target_column' in experiment['meta']:
            #     del experiment['meta']['target_column']
            # Add target_positiveclass to parameters
            experiment['parameters']['target_positiveclass'] = source_dataset['target_positiveclass']
            # Add mappings to parameters
            if 'fair_column_mappings' in source_dataset:
                experiment['parameters']['fair_column_mappings'] = source_dataset['fair_column_mappings']
            
        with open(exp_item, 'w') as file:
            json.dump(experiment, file)
    print('Done')