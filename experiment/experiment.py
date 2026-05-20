import logging
import os
from clearbox_engine import Dataset, Preprocessor, TabularEngine
import time
import pandas as pd
from data_framework.imbalance.imbalance import Imbalance
# from data_framework.data_quality.quality import Quality
from clearbox_engine.engine.tabular_engine_simpsonized import TabularEngineSimpsonized
from clearbox_engine.engine.tabular_engine_term import TabularEngineTerm
from clearbox_engine.engine.tabular_engine_weight import TabularEngineWeight
import json
from sklearn.model_selection import train_test_split
import numpy as np

class Experiment:
    """
    Represents an experiment for training a model on a dataset.
    Args:
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs for training.
        learning_rate (float): The learning rate for training.
        decay (float): The decay rate for training.
        simpson_weight (float): The weight for Simpsonization.
        simpson_magnitude (float): The magnitude for Simpsonization.
        simpson_exp_magnitude (float): The exponential magnitude for Simpsonization.
        exp_id (str, optional): The ID of the experiment. Defaults to None.
        only_categorical (bool, optional): Whether to use only categorical features. Defaults to False.
        col_subset (Any, optional): The subset of columns to use. Defaults to None.
        base_folder (str, optional): The base folder for the experiment. Defaults to None.
        source_dataset (dict, optional): The source dataset information. Defaults to None.
    Attributes:
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs for training.
        learning_rate (float): The learning rate for training.
        decay (float): The decay rate for training.
        simpson_weight (float): The weight for Simpsonization.
        simpson_magnitude (float): The magnitude for Simpsonization.
        simpson_exp_magnitude (float): The exponential magnitude for Simpsonization.
        time (str): The timestamp of the experiment.
        only_categorical (bool): Whether to use only categorical features.
        experiment_base_path (str): The base path for the experiment dataset.
        experiment_dataset_name (str): The name of the experiment dataset.
        experiment_dataset_path (str): The path to the experiment dataset.
        experiment_output_path (str): The output path for the experiment.
        experiment_code (str): The code for the experiment.
        synthetic_dataset_filename (str): The filename of the synthetic dataset.
        path_to_json (str): The path to the experiment JSON file.
        source_dataset (dict): The source dataset information.
        logger (Logger): The logger for the experiment.
    Methods:
        get_input_params(): Returns the input parameters of the experiment.
        run(): Runs the experiment.
    """
    def __init__(self, loss_model: str, batch_size: int, epochs: int, learning_rate: float, decay: float, 
                 simpson_weight: float, simpson_magnitude: float, simpson_exp_magnitude: float,
                 exp_id: str = None, only_categorical: bool = False, layers_size: list = [50], noy: bool = False, 
                 split_seed:int = 42, col_subset = None, base_folder: str = None, source_dataset: dict = None):
        self.loss_model = loss_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay = decay
        self.simpson_weight = simpson_weight
        self.simpson_magnitude = simpson_magnitude
        self.simpson_exp_magnitude = simpson_exp_magnitude
        if exp_id is None:
            self.time = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        else:
            self.time = exp_id
        self.only_categorical = only_categorical
        self.layers_size = layers_size
        self.noy = noy
        self.split_seed = split_seed
        learning_rate_str = str(self.learning_rate).replace(".", "_")
        simpson_weight_str = str(self.simpson_weight).replace(".", "_")
        self.dataset_name = source_dataset["name"]
        if base_folder is None:
            raise ValueError("base_folder cannot be None. Please provide a valid base_folder.")
        else:
            self.experiment_base_path = base_folder.split('/')[0]
            self.experiment_dataset_name = base_folder.split('/')[1] # 01_adult
            self.experiment_dataset_path = base_folder  # synthetic-datasets/01_adult/YYMMGG-HHMMSS-R-C/
        self.experiment_output_path = self.experiment_dataset_path + self.time + "/"  # synthetic-datasets/01_adult/YYMMGG-HHMMSS-R-C/YYMMGG-HHMMSS/
        self.experiment_output_path_datasets = self.experiment_output_path + "datasets/"
        self.experiment_output_path_balance = self.experiment_output_path + "balance/"
        self.experiment_output_path_states = self.experiment_output_path + "states/"
        if not os.path.exists(self.experiment_output_path_states):
            os.makedirs(self.experiment_output_path_states)
        self.experiment_output_path_viz = self.experiment_output_path + "viz/"
        self.experiment_code = f"{self.experiment_dataset_name}_{self.time}" # 01_adult_YYMMGG-HHMMSS
        self.experiment_code_sd = f"{self.experiment_dataset_name}_{self.time}_sd" # 01_adult_YYMMGG-HHMMSS_sd
        self.experiment_code_od = f"{self.experiment_dataset_name}_{self.time}_od" # 01_adult_YYMMGG-HHMMSS_od
        self.synthetic_dataset_filename = self.experiment_output_path_datasets + self.experiment_code_sd + ".csv" # synthetic-datasets/01_adult/YYMMGG-HHMMSS-R-C/YYMMGG-HHMMSS/datasets/01_adult_YYMMGG-HHMMSS_sd.csv
        self.path_to_json = f'{self.experiment_output_path}experiment.json'
        if source_dataset is None:
            raise ValueError("source_dataset cannot be None. Please provide a valid source_dataset.")
        else:
            self.source_dataset = source_dataset
        self.logger = logging.getLogger(f'engine.experiment.Experiment')
        self.logger.info(self.get_input_params())

        # check if output directories exists, if not create it
        if not os.path.exists(self.experiment_output_path):
            os.makedirs(self.experiment_output_path)
        if not os.path.exists(self.experiment_output_path_datasets):
            os.makedirs(self.experiment_output_path_datasets)
        if not os.path.exists(self.experiment_output_path_balance):
            os.makedirs(self.experiment_output_path_balance)
        self.logger.info(f'synthetic dataset in {self.synthetic_dataset_filename}')
        print("* " * 40)
        print(f"{self.get_input_params()}\n")

    def get_input_params(self):
        return {'loss_model': self.loss_model, 'epochs': self.epochs, 'batch_size': self.batch_size, 'learning_rate': self.learning_rate, 'decay': self.decay,
                'simpson_weight': self.simpson_weight, 'simpson_magnitude': self.simpson_magnitude, 'simpson_exp_magnitude': self.simpson_exp_magnitude, 
                'time': self.time, 'only_categorical': self.only_categorical, 'layers_size': self.layers_size, 'noy':self.noy, 'target_column': self.source_dataset['target_column'], 'target_positiveclass': self.source_dataset['target_positiveclass'], 'split_seed': self.split_seed,
                'sensitive_features': self.source_dataset["sensitive_features"], 'fair_column_mappings': self.source_dataset["fair_column_mappings"], 
                'sep': self.source_dataset["sep"] if "sep" in self.source_dataset else None,
                'num_rows': self.source_dataset["num_rows"] if "num_rows" in self.source_dataset else None,
                'header': self.source_dataset["header"] if "header" in self.source_dataset else None,
                'names': self.source_dataset["names"] if "names" in self.source_dataset else None,
                'dtype': self.source_dataset["dtype"] if "dtype" in self.source_dataset else None,
                'usecols': self.source_dataset["usecols"] if "usecols" in self.source_dataset else None,
                'path_to_json': self.path_to_json}
                
    def usecols_to_columns(self, params):
        columns_field = ""
        if "cols_names" in params:
            columns_field = "cols_names"
        elif "names" in params:
            columns_field = "names"
        else:
            raise ValueError("No columns field found in the parameters")
        if "usecols" in params:
            params[columns_field] = params["usecols"]
        return params
    
    def execute_preprocess(self, type, read_csv_params, from_csv_params, to_csv_params, path):
        self.logger.debug(f"Source {type} dataset need preprocessing")
        self.logger.debug(f"read_csv_parameters_{type}: {read_csv_params}")
        dataset = pd.read_csv(**read_csv_params)
        dataset = self.source_dataset["preprocess"](dataset)
        # Reorder columns to match usecols order before saving TO BE TESTED
        if 'usecols' in self.source_dataset:
            dataset = dataset[self.source_dataset['usecols']]
        preprocessed_to_csv_parameters = to_csv_params.copy()
        # path[type]=path[type][:path[type].rfind('.')] + "_PREPROCESSED.csv"
        path[type] = self.experiment_output_path_datasets + self.experiment_code_od + "_PREPROCESSED.csv"
        preprocessed_to_csv_parameters["path_or_buf"]=path[type]
        dataset.to_csv(**preprocessed_to_csv_parameters)
        from_csv_params['csv_file'] = path[type]
        read_csv_params['filepath_or_buffer'] = path[type]
        from_csv_params = self.usecols_to_columns(from_csv_params)
        read_csv_params = self.usecols_to_columns(read_csv_params)
        self.logger.debug(f"Source {type} dataset preprocessing done")
        return read_csv_params, from_csv_params, path

    def check_only_categorical(self, read_csv, from_csv):
        if self.only_categorical:
            self.logger.info(f"Only categorical features will be used")
            read_csv['usecols'] = self.source_dataset["categorical_columns"]
            from_csv['usecols'] = self.source_dataset["categorical_columns"]
        else:
            self.logger.info(f"All features will be used")
        return read_csv, from_csv

    def execute_split(self, path, read_csv_train_params, from_csv_train_params, to_csv_params, random_state):
        self.logger.info(f"Splitting {path['train']} into train and validation datasets")
        data = pd.read_csv(**read_csv_train_params)
        train_dataset, validation_dataset = train_test_split(data, test_size=0.2, random_state=random_state)
        # Reorder columns to match usecols order before saving TO BE TESTED
        if 'usecols' in self.source_dataset:
            train_dataset = train_dataset[self.source_dataset['usecols']]
            validation_dataset = validation_dataset[self.source_dataset['usecols']]
        actual_train_path = path['train']
        # path['train'] = actual_train_path[:actual_train_path.rfind('.')] + "_TRAIN.csv"
        path['train'] = self.experiment_output_path_datasets + self.experiment_code_od + "_TRAIN.csv"
        self.logger.debug(f"new train path: {path['train']}")
        # path['test'] = actual_train_path[:actual_train_path.rfind('.')] + "_TEST.csv"
        path['test'] = self.experiment_output_path_datasets + self.experiment_code_od + "_TEST.csv"
        self.logger.debug(f"new test path: {path['test']}")
        splitted_to_csv_parameters = to_csv_params.copy()
        splitted_to_csv_parameters["path_or_buf"]=path['train']
        train_dataset.to_csv(**splitted_to_csv_parameters)
        splitted_to_csv_parameters["path_or_buf"]=path['test']
        validation_dataset.to_csv(**splitted_to_csv_parameters)
        from_csv_train_params['csv_file'] = path['train']
        read_csv_train_params['filepath_or_buffer']=path['train']
        from_csv_train_params = self.usecols_to_columns(from_csv_train_params)
        read_csv_train_params = self.usecols_to_columns(read_csv_train_params)
        return path, read_csv_train_params, from_csv_train_params
    
    def initialize_csv_parameters(self):
        # from_csv for Dataset.from_csv: _train and _test
        # read_csv for pd.read_csv: _train and _test

        # TRAIN > FROM_CSV
        path = {'train': self.source_dataset["train_path"], 'test': self.source_dataset["validation_path"], 'synth': self.synthetic_dataset_filename}
        from_csv_parameters_train = {}
        from_csv_parameters_train['csv_file']=self.source_dataset["train_path"]
        if not self.noy:
            from_csv_parameters_train['target_column']=self.source_dataset["target_column"]
        from_csv_parameters_train['regression']=self.source_dataset["regression"]
        if "names" in self.source_dataset:
            from_csv_parameters_train['cols_names']=self.source_dataset["names"]
        if "dtype" in self.source_dataset:
            from_csv_parameters_train['dtype']=self.source_dataset["dtype"]
        if "sep" in self.source_dataset:
            from_csv_parameters_train['sep'] = self.source_dataset["sep"]
        if "na_values" in self.source_dataset:
            from_csv_parameters_train['na_values'] = self.source_dataset["na_values"]
        if "header" in self.source_dataset:
            from_csv_parameters_train['header'] = self.source_dataset["header"]
        if "usecols" in self.source_dataset:
            from_csv_parameters_train['usecols'] = self.source_dataset["usecols"]

        # TRAIN > READ_CSV
        read_csv_parameters_train = from_csv_parameters_train.copy()
        if "cols_names" in read_csv_parameters_train:
            read_csv_parameters_train['names']=read_csv_parameters_train["cols_names"]
            del read_csv_parameters_train["cols_names"]
        if "parse_dates" in self.source_dataset:
            read_csv_parameters_train['parse_dates']=self.source_dataset["parse_dates"]
        if "date_format" in self.source_dataset:
            read_csv_parameters_train['date_format']=self.source_dataset["date_format"]
        read_csv_parameters_train['filepath_or_buffer']=from_csv_parameters_train['csv_file']
        if not self.noy:
            del read_csv_parameters_train["target_column"]
        del read_csv_parameters_train["regression"]
        del read_csv_parameters_train["csv_file"]

        # TO_CSV > TRAIN

        # TO_CSV > SYNTH
        to_csv_parameters_synth={}
        to_csv_parameters_synth["path_or_buf"]=self.synthetic_dataset_filename
        to_csv_parameters_synth["index"]=False
        if "sep" in from_csv_parameters_train:
            to_csv_parameters_synth["sep"]=from_csv_parameters_train["sep"]
        if "header" in from_csv_parameters_train and from_csv_parameters_train["header"] is None:
            to_csv_parameters_synth["header"]=False
        if "date_format" in read_csv_parameters_train:
            to_csv_parameters_synth["date_format"]=read_csv_parameters_train["date_format"]
        self.logger.debug(f"to_csv_parameters_synth: {to_csv_parameters_synth}")
        # Check if preprocessing and/or splitting is needed
        if ("preprocess" in self.source_dataset) and (self.source_dataset["validation_path"] is not None):
            self.logger.info("[I] Source train and validation need preprocessing")
            read_csv_parameters_train, from_csv_parameters_train, path = self.execute_preprocess(type="train", read_csv_params=read_csv_parameters_train, from_csv_params=from_csv_parameters_train, to_csv_params=to_csv_parameters_synth, path=path)
            read_csv_parameters_test, from_csv_parameters_test, path = self.execute_preprocess(type="test", read_csv_params=read_csv_parameters_test, from_csv_params=from_csv_parameters_test, to_csv_params=to_csv_parameters_synth, path=path)          
        elif ("preprocess" in self.source_dataset) and (self.source_dataset["validation_path"] is None):
            self.logger.info("[II] Source train need preprocessing and needs to be splitted")
            read_csv_parameters_train, from_csv_parameters_train, path = self.execute_preprocess(type="train", read_csv_params=read_csv_parameters_train, from_csv_params=from_csv_parameters_train, to_csv_params=to_csv_parameters_synth, path=path)
            path, read_csv_parameters_train, from_csv_parameters_train = self.execute_split(path=path, read_csv_train_params=read_csv_parameters_train, from_csv_train_params=from_csv_parameters_train, to_csv_params=to_csv_parameters_synth, random_state=self.split_seed)
        elif ("preprocess" not in self.source_dataset) and (self.source_dataset["validation_path"] is not None):
            self.logger.info("[III] No preprocessing and no splitting needed")
        elif ("preprocess" not in self.source_dataset) and (self.source_dataset["validation_path"] is None):
            self.logger.info("[IV] No preprocessing needed, splitting needed")
            path, read_csv_parameters_train, from_csv_parameters_train = self.execute_split(path=path, read_csv_train_params=read_csv_parameters_train, from_csv_train_params=from_csv_parameters_train, to_csv_params=to_csv_parameters_synth, random_state=self.split_seed)
        
        self.check_only_categorical(read_csv_parameters_train, from_csv_parameters_train)

        self.logger.debug(f"path: {path}")
        self.logger.debug(f"from_csv_parameters_train: {from_csv_parameters_train}")
        self.logger.debug(f"read_csv_parameters_train: {read_csv_parameters_train}")

        # TEST > FROM_CSV
        from_csv_parameters_test = from_csv_parameters_train.copy()
        from_csv_parameters_test['csv_file']=path['test']
        self.logger.debug(f"from_csv_parameters_test: {from_csv_parameters_test}")

        # TEST > READ_CSV
        read_csv_parameters_test = read_csv_parameters_train.copy()
        read_csv_parameters_test['filepath_or_buffer']=path['test']
        self.logger.debug(f"read_csv_parameters_test: {read_csv_parameters_test}")

        # SYNTH > FROM_CSV
        from_csv_parameters_synth = from_csv_parameters_train.copy()
        from_csv_parameters_synth['csv_file']=self.synthetic_dataset_filename
        from_csv_parameters_synth = self.usecols_to_columns(from_csv_parameters_synth)
        self.logger.debug(f"from_csv_parameters_synth: {from_csv_parameters_synth}")

        # SYNTH > READ_CSV
        read_csv_parameters_synth = read_csv_parameters_train.copy()
        read_csv_parameters_synth['filepath_or_buffer']=self.synthetic_dataset_filename
        read_csv_parameters_synth = self.usecols_to_columns(read_csv_parameters_synth)
        self.logger.debug(f"read_csv_parameters_synth: {read_csv_parameters_synth}")

        parameters = {
            "from_csv": {
                "train": from_csv_parameters_train,
                "test": from_csv_parameters_test,
                "synth": from_csv_parameters_synth
            },
            "read_csv": {
                "train": read_csv_parameters_train,
                "test": read_csv_parameters_test,
                "synth": read_csv_parameters_synth
            },
            "to_csv": {
                "synth": to_csv_parameters_synth
            }
        }

        return parameters

    def create_balance_mask(self, categorical_features):
        balance_mask = []
        for feature in categorical_features:
            if feature in self.source_dataset["sensitive_features"]:
                balance_mask.append(1)
            else:
                balance_mask.append(0)
        return balance_mask

    def run(self):
        parameters = self.initialize_csv_parameters()
        train_dataset = Dataset.from_csv(**parameters["from_csv"]["train"])
        validation_dataset = Dataset.from_csv(**parameters["from_csv"]["test"])

        # def _infer_feature_types(dataset: Dataset) -> tuple
        # Check each column if it is ordinal (number, date), categorical (string, boolean) or datetime (datetime)
        # train_dataset = train_dataset[sorted(train_dataset.columns)]
        preprocessor = Preprocessor(train_dataset)
        self.logger.info(f'Preprocessor created')
        self.logger.info(f"First row of the dataframe: {train_dataset.data.iloc[0]}")
        self.logger.info(f"train_dataset.columns(): {train_dataset.columns()}")
        self.logger.info(f"train_dataset.dtypes: {train_dataset.columns_types()}")
        self.logger.info(f"preprocessor.get_categorical_features(): {preprocessor.get_categorical_features()}")
        self.logger.info(f"preprocessor.get_features_sizes()[1]: {preprocessor.get_features_sizes()[1]}")
        # print(f"train_dataset.columns(): {train_dataset.columns()}")
        # print(f"preprocessor.get_categorical_features(): {preprocessor.get_categorical_features()}")
        # print(f"preprocessor.get_features_sizes()[1]: {preprocessor.get_features_sizes()[1]}")
        self.logger.info(f'sensitive_features: {self.source_dataset["sensitive_features"]}')
        self.balance_mask = self.create_balance_mask(categorical_features=preprocessor.get_categorical_features())
        self.logger.info(f'balance_mask: {self.balance_mask}')

        train_ds = preprocessor.transform(train_dataset.get_x())
        self.logger.info(f"train_ds[0]: {train_ds[0]}")
        self.logger.info(f'train_ds.shape: {train_ds.shape}')
        validation_ds = preprocessor.transform(validation_dataset.get_x())
        self.logger.info(f'validation_ds.shape: {validation_ds.shape}')
        if not self.noy:
            if train_dataset.regression:
                Y = train_dataset.get_normalized_y()
            else:
                Y = train_dataset.get_one_hot_encoded_y()
        features_size = train_ds.shape[1]
        self.logger.info(f'features_size: {features_size}')
        # Instantiate the engine
        engine_parameters ={}
        engine_parameters["layers_size"] = self.layers_size
        engine_parameters["x_shape"] = train_ds[0].shape
        engine_parameters["ordinal_feature_sizes"] = preprocessor.get_features_sizes()[0]
        engine_parameters["categorical_feature_sizes"] = preprocessor.get_features_sizes()[1]
        if not self.noy:
            engine_parameters["y_shape"] = Y[0].shape
        engine_parameters["train_params"] = {
                    "l2_reg": 0.000,
                    "beta": 0,
                    "alpha": .1,
                    "gauss_s": 0.01,
                    "gauss_s_c": 0.1,
                    "weight_decay": self.decay,
                    "prob_clip": 0.99,
                    "simpson_weight": self.simpson_weight,
                    "simpson_magnitude": self.simpson_magnitude,
                    "simpson_exp_magnitude": self.simpson_exp_magnitude,
                    "balance_mask": self.balance_mask
                }
        engine_parameters["folder_path"] = self.experiment_output_path
        if self.loss_model == 'vanilla':
            engine = TabularEngine(**engine_parameters)
        elif self.loss_model == 'simpsonized':
            engine = TabularEngineSimpsonized(**engine_parameters)
        elif self.loss_model == 'term':
            engine = TabularEngineTerm(**engine_parameters)
        elif self.loss_model == 'weight':
            engine = TabularEngineWeight(**engine_parameters)
        else:
            raise ValueError("loss_model not recognized")
        tic = {}
        toc = {}
        tic['fit'] = time.perf_counter()
        if self.noy:
            engine.fit(train_ds, batch_size=self.batch_size, epochs=self.epochs,learning_rate=self.learning_rate)
        else:
            engine.fit(train_ds, y_train_ds=Y, batch_size=self.batch_size, epochs=self.epochs,learning_rate=self.learning_rate)
        toc['fit'] = time.perf_counter()
        self.logger.info(f'Engine fitted')
        engine.save(f"{self.experiment_output_path_states}architecture.txt", f"{self.experiment_output_path_states}state-dict.txt")
        self.logger.info('Engine saved')
        self.logger.debug(f'engine.architecture: {engine.architecture}')
        # print(engine.architecture)
        # print(engine.params)

        self.logger.info(f'saving latent space')
        b1, b2, b3 = engine.apply(train_ds) #b2 distribuzione punti spazio latenti b1: ricostruzione b3: varianza spazio latente
        np.save(f'{self.experiment_output_path_states}b1_reconstruction.npy', b1)
        np.save(f'{self.experiment_output_path_states}b2_latentspace_distribution.npy', b2)
        np.save(f'{self.experiment_output_path_states}b3_latentspace_variance.npy', b3)
        self.logger.info('b1 shape: {}'.format(b1.shape))
        # eventualmente passargli anche Y come secondo parametro
        print(f"b2.shape: {b2.shape}")


        from clearbox_engine import LabeledSynthesizer, UnlabeledSynthesizer
        tic['synthesizer'] = time.perf_counter()
        if self.noy:
            self.logger.debug(f'Starting Unlabeled Synthesizer...')
            synthesizer = UnlabeledSynthesizer(train_dataset, engine, preprocessor)
        else:
            self.logger.debug(f'Starting Labeled Synthesizer...')
            synthesizer = LabeledSynthesizer(train_dataset, engine, preprocessor)
        pd_synthetic_dataset = synthesizer.generate(has_header=True)
        toc['synthesizer'] = time.perf_counter()
        pd_synthetic_dataset.to_csv(**parameters["to_csv"]["synth"])
        synthetic_dataset = Dataset.from_csv(**parameters["from_csv"]["synth"])
        self.logger.debug(f'Synthetic labels generated in {toc["synthesizer"]-tic["synthesizer"]} s.')

        #TODO Implement a progress bar for the following operations

        #NOTE RECONSTRUCTION ERROR
        from clearbox_engine import ReconstructionError
        self.logger.debug('Starting reconstruction error...')
        tic['reconstruction_error'] = time.perf_counter()
        re = ReconstructionError(train_dataset, synthetic_dataset, engine, preprocessor).get()
        toc['reconstruction_error'] = time.perf_counter()
        self.logger.debug(f'...reconstruction error done in {toc["reconstruction_error"]-tic["reconstruction_error"]} s.')
        self.logger.info(f'Reconstruction error:\n{re}')

        from clearbox_engine import PrivacyScore
        self.logger.debug('Starting privacy score...')
        tic['privacy_score'] = time.perf_counter()
        privacy_metrics = PrivacyScore(train_dataset, synthetic_dataset, validation_dataset, preprocessor, parallel=False).get()
        toc['privacy_score'] = time.perf_counter()
        self.logger.debug(f'...privacy score done in {toc["privacy_score"]-tic["privacy_score"]} s.')
        training_metrics = privacy_metrics['training_metrics']
        synthetic_metrics = privacy_metrics['synthetic_metrics']
        synthetic_training_metrics = privacy_metrics['synthetic_training_metrics']
        synthetic_holdout_metrics = privacy_metrics['synthetic_holdout_metrics']
        membership_inference_test = privacy_metrics['membership_inference_test']

        self.logger.info(f"\n== TRAINING METRICS\n" +
                         "=== Duplicates in Training Dataset: {}\n".format(training_metrics['training_duplicates']) +
                         "=== Duplicates percentage in Training Dataset: {:.2f}%\n".format(
                             training_metrics['training_duplicates_percentage']) +
                         "=== Unique Duplicates in Training Dataset: {}\n".format(
                             training_metrics['training_unique_duplicates']) +
                         "=== Unique Duplicates percentage in Training Dataset: {:.2f}%\n".format(
                             training_metrics['training_unique_duplicates_percentage']) +
                         "\n=== Training Dataset DCR Stats: {}\n".format(training_metrics['dcr_train_train_stats'])
                         )
        # print("== TRAINING METRICS")
        # print("=== Duplicates in Training Dataset: {}".format(training_metrics['training_duplicates']))
        # print(
        #     "=== Duplicates percentage in Training Dataset: {:.2f}%".format(
        #         training_metrics['training_duplicates_percentage']))
        # print("=== Unique Duplicates in Training Dataset: {}".format(training_metrics['training_unique_duplicates']))
        # print("=== Unique Duplicates percentage in Training Dataset: {:.2f}%".format(
        #     training_metrics['training_unique_duplicates_percentage']))
        # print("\n=== Training Dataset DCR Stats:")
        # for k, v in training_metrics['dcr_train_train_stats'].items():
        #     print("==== {}: {}".format(k, v))
        # print("\n=== Training Dataset DCR Hist:")
        # plt.figure(figsize=(26, 12))
        # plt.xticks(rotation='vertical')
        # plt.bar(training_metrics['dcr_train_train_hist']['bins'], training_metrics['dcr_train_train_hist']['counts'])

        # In[22]:
        self.logger.info("\n==SYNTHETIC METRICS\n"
                         "=== Duplicates in Synthetic Dataset: {}\n".format(
            synthetic_metrics['synthetic_duplicates']) +
                         "=== Duplicates percentage in Synthetic Dataset: {:.2f}%\n".format(
                             synthetic_metrics['synthetic_duplicates_percentage']) +
                         "=== Unique Duplicates in Synthetic Dataset: {}\n".format(
                             synthetic_metrics['synthetic_unique_duplicates']) +
                         "=== Unique Duplicates percentage in Synthetic Dataset: {:.2f}%\n".format(
                             synthetic_metrics['synthetic_unique_duplicates_percentage'])
                         )
        # print("== SYNTHETIC METRICS")
        # print("=== Duplicates in Synthetic Dataset: {}".format(synthetic_metrics['synthetic_duplicates']))
        # print("=== Duplicates percentage in Synthetic Dataset: {:.2f}%".format(
        #     synthetic_metrics['synthetic_duplicates_percentage']))
        # print("=== Unique Duplicates in Synthetic Dataset: {}".format(synthetic_metrics['synthetic_unique_duplicates']))
        # print("=== Unique Duplicates percentage in Synthetic Dataset: {:.2f}%".format(
        #     synthetic_metrics['synthetic_unique_duplicates_percentage']))

        self.logger.info("\n== SYNTHETIC-TRAINING METRICS\n"
                         "=== Clones from Training in Synthetic Dataset: {}\n".format(
            synthetic_training_metrics['synth_train_clones']) +
                         "=== Clones percentage in Synthetic Dataset: {:.2f}%\n".format(
                             synthetic_training_metrics['synth_train_clones_percentage']) +
                         "\n=== Synthetic-Training DCR Stats: {}".format(
                             synthetic_training_metrics['dcr_synth_train_stats'])
                         )
        # print("== SYNTHETIC-TRAINING METRICS")
        # print("=== Clones from Training in Synthetic Dataset: {}".format(synthetic_training_metrics['synth_train_clones']))
        # print("=== Clones percentage in Synthetic Dataset: {:.2f}%".format(
        #     synthetic_training_metrics['synth_train_clones_percentage']))
        # print("\n=== Synthetic-Training DCR Stats:")
        # for k, v in synthetic_training_metrics['dcr_synth_train_stats'].items():
        #     print("==== {}: {}".format(k, v))
        # print("\n=== Synthetic-Training DCR Hist:")
        # plt.figure(figsize=(26, 12))
        # plt.xticks(rotation='vertical')
        # plt.bar(synthetic_training_metrics['dcr_synth_train_hist']['bins'],
        #         synthetic_training_metrics['dcr_synth_train_hist']['counts'])
        # print("\n=== Training vs Synthetic-Training DCR Hist:")
        # labels = training_metrics['dcr_train_train_hist']['bins']
        # train_train_counts = training_metrics['dcr_train_train_hist']['counts']
        # synth_train_counts = synthetic_training_metrics['dcr_synth_train_hist']['counts']
        # X_axis = np.arange(len(labels))
        # plt.figure(figsize=(26, 12))
        # plt.xticks(X_axis, labels, rotation='vertical')
        # plt.bar(X_axis - 0.2, train_train_counts, 0.4, color='r', label='Train-Train')
        # plt.bar(X_axis + 0.2, synth_train_counts, 0.4, color='g', label='Synth-Train')

        # In[24]:
        self.logger.info("\n== SYNTHETIC-HOLDOUT METRICS\n"
                         "=== Synthetic-Holdout DCR Stats: {}\n".format(
            synthetic_holdout_metrics['dcr_synth_holdout_stats']) +
                         "=== Synthetic-Training vs Synthetic-Holdout Test: {:.2f}\n".format(
                             synthetic_holdout_metrics['synth_holdout_test'])
                         )
        # print("== SYNTHETIC-HOLDOUT METRICS")
        # print("\n=== Synthetic-Holdout DCR Stats:")
        # for k, v in synthetic_holdout_metrics['dcr_synth_holdout_stats'].items():
        #     print("==== {}: {}".format(k, v))
        # print("\n=== Synthetic-Training vs Synthetic-Holdout Test: {:.2f}".format(
        #     synthetic_holdout_metrics['synth_holdout_test']))
        # print("\n=== Synthetic-Training vs Synthetic-Holdout DCR Hist:")
        # labels = synthetic_holdout_metrics['dcr_synth_holdout_hist']['bins']
        # synth_holdout_counts = synthetic_holdout_metrics['dcr_synth_holdout_hist']['counts']
        # if 'dcr_synth_train_hist' in synthetic_holdout_metrics:
        #     synth_train_counts = synthetic_holdout_metrics['dcr_synth_train_hist']['counts']
        # else:
        #     synth_train_counts = synthetic_training_metrics['dcr_synth_train_hist']['counts']
        # X_axis = np.arange(len(labels))
        # plt.figure(figsize=(26, 12))
        # plt.xticks(X_axis, labels, rotation='vertical')
        # plt.bar(X_axis - 0.2, synth_train_counts, 0.4, color='r', label='Synth-Train')
        # plt.bar(X_axis + 0.2, synth_holdout_counts, 0.4, color='g', label='Synth-Holdout')

        # In[25]:

        self.logger.info("\n== MEMBERSHIP INFERENCE TEST\n"
                         "=== Adversary Distance Thresholds: {}\n".format(
            membership_inference_test['adversary_distance_thresholds']) +
                         "=== Adversary Precision Score: {}\n".format(
                             membership_inference_test['adversary_precisions']) +
                         "=== Membership Inference Mean Risk Score: {}\n".format(
                             membership_inference_test['membership_inference_mean_risk_score'])
                         )
        # print("== MEMBERSHIP INFERENCE TEST")
        # print("=== Adversary Distance Thresholds: {}".format(membership_inference_test['adversary_distance_thresholds']))
        # print("=== Adversary Precision Score: {}".format(membership_inference_test['adversary_precisions']))
        # print("=== Membership Inference Mean Risk Score: {}".format(
        #     membership_inference_test['membership_inference_mean_risk_score']))

        #NOTE DETECTION SCORE
        from clearbox_engine import DetectionScore
        self.logger.debug('Starting DetectionScore...')
        tic['detection_score'] = time.perf_counter()
        detection_score = DetectionScore(train_dataset, synthetic_dataset, preprocessor).get()
        toc['detection_score'] = time.perf_counter()
        self.logger.debug(f'Detection Score done in {toc["detection_score"]-tic["detection_score"]} s.')
        self.logger.info(f'Detection Score:\n{detection_score}')

        #NOTE MUTUAL INFORMATION
        from clearbox_engine import MutualInformation
        self.logger.debug('Starting Mutual Information...')
        tic['mutual_information'] = time.perf_counter()
        mi = MutualInformation(train_dataset, synthetic_dataset).get()
        toc['mutual_information'] = time.perf_counter()
        self.logger.debug(f'Mutual Information done in {toc["mutual_information"]-tic["mutual_information"]} s.')
        self.logger.info(f'Original mutual information: {mi["original_mutual_information"]}')
        # for corr in mi['original_mutual_information']:
        #     for value in corr:
        #         print("{:.4f}".format(value), end='\t\t')
        #     print()
        #     print()

        #NOTE FEATURES COMPARISON
        from clearbox_engine import FeaturesComparison
        self.logger.debug('Starting Features Comparison...')
        tic['features_comparison'] = time.perf_counter()
        fc = FeaturesComparison(train_dataset, synthetic_dataset, preprocessor).get()
        toc['features_comparison'] = time.perf_counter()
        self.logger.debug(f'Features comparison done in {toc["features_comparison"]-tic["features_comparison"]} s.')
        self.logger.info(f'Features comparison:\n{fc}')

        #NOTE QUERY POWER
        from clearbox_engine import QueryPower
        self.logger.debug('Starting Query Power...')
        tic['query_power'] = time.perf_counter()
        qp = QueryPower(train_dataset, synthetic_dataset, preprocessor).get()
        toc['query_power'] = time.perf_counter()
        self.logger.debug(f'Query power done in {toc["query_power"]-tic["query_power"]} s.')
        self.logger.info(f'Query power:\n{qp}')

        #NOTE IMBALANCE
        self.logger.debug('Starting Imbalance...')
        tic['imbalance'] = time.perf_counter()
        imbalance_features = self.source_dataset["imbalance_features"]
        dataset = pd.read_csv(**parameters["read_csv"]["train"])
        original_imbalance = Imbalance(f'{self.experiment_dataset_name}_DB-0-original', dataset, imbalance_features,
                                       output_path=self.experiment_output_path_balance, verbose=False)
        original_imbalance.frequencies(to_file=True, to_console=False)
        # Quality(f'{experiment_dataset_name}_DQ-0-original', train_dataset_path, output_path=experiment_dataset_path, isurl=False)

        dataset = pd.read_csv(**parameters["read_csv"]["synth"])
        synthetic_imbalance = Imbalance(f'{self.experiment_code_sd}_DB-1-synthetic', dataset, imbalance_features,
                                        output_path=self.experiment_output_path_balance, verbose=False)
        synthetic_imbalance.frequencies(to_file=True, to_console=False)
        # Quality(f'{experiment_code}_DQ-1-synthetic', synthetic_dataset_filename, output_path=experiment_output_path, isurl=False)
        toc['imbalance'] = time.perf_counter()
        self.logger.debug(f'Imbalance done in {toc["imbalance"]-tic["imbalance"]} s.')

        if self.noy:
            self.logger.debug(f"New dataset import to reinclude target column")
            parameters["from_csv"]["train"]["target_column"]=self.source_dataset["target_column"]
            self.logger.debug(f"from_csv_parameters_train: {parameters['from_csv']['train']}")
            parameters["from_csv"]["synth"]["target_column"]=self.source_dataset["target_column"]
            self.logger.debug(f"from_csv_parameters_train: {parameters['from_csv']['train']}")
            parameters["from_csv"]["test"]["target_column"]=self.source_dataset["target_column"]
            self.logger.debug(f"from_csv_parameters_train: {parameters['from_csv']['train']}")
            train_dataset = Dataset.from_csv(**parameters["from_csv"]["train"])
            validation_dataset = Dataset.from_csv(**parameters["from_csv"]["test"])
            preprocessor = Preprocessor(train_dataset)
            synthetic_dataset = Dataset.from_csv(**parameters["from_csv"]["synth"])      

        #NOTE TSTR SCORE
        from clearbox_engine import TSTRScore
        self.logger.debug('Starting TSTRScore...')
        tic['tstr_score'] = time.perf_counter()
        TSTR_score = TSTRScore(original_dataset = train_dataset, synthetic_dataset=synthetic_dataset, validation_dataset=validation_dataset, preprocessor=preprocessor).get()
        toc['tstr_score'] = time.perf_counter()
        self.logger.debug(f'TSTRScore done in {toc["tstr_score"]-tic["tstr_score"]} s.')
        self.logger.info(f'TSTR Score:\n{TSTR_score}')

        #NOTE other statistics
        stats = {}
        stats['train'] = {
            'num_records': train_dataset.data.shape[0],
            'target_column': self.source_dataset["target_column"],
            'target_column_counts': train_dataset.data[self.source_dataset["target_column"]].value_counts().to_dict()
        }
        stats['synthetic'] = {
            'num_records': synthetic_dataset.data.shape[0],
            'target_column': self.source_dataset["target_column"],
            'target_column_counts': synthetic_dataset.data[self.source_dataset["target_column"]].value_counts().to_dict()
        }

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Imbalance):
                    return obj.default(obj)
                return super().default(obj)

        results = {
            "training_metrics": training_metrics,
            "synthetic_metrics": synthetic_metrics,
            "synthetic_holdout_metrics": synthetic_holdout_metrics,
            "synthetic_training_metrics": synthetic_training_metrics,
            "detection_score": detection_score,
            "tstr": TSTR_score,
            "mutual_information": mi,
            "feature_comparison": fc,
            "query_power": qp,
            "reconstruction_error": re,
            "training_imbalance": original_imbalance,
            "synthetic_imbalance": synthetic_imbalance
        }

        experiment = {
            "parameters": self.get_input_params(),
            "results": results,
            "meta": {'fit_time': (toc['fit'] - tic['fit']),
                     'dataset_name': self.dataset_name},
            "path": {
                "train": parameters["from_csv"]["train"]["csv_file"],
                "test": parameters["from_csv"]["test"]["csv_file"],
                "synth": parameters["from_csv"]["synth"]["csv_file"],
                "synth_path": self.synthetic_dataset_filename,
                "states": self.experiment_output_path_states,
                "balance": self.experiment_output_path_balance,
                "viz": self.experiment_output_path_viz,
                "datasets": self.experiment_output_path_datasets
            }
        }

        self.logger.debug("saving experiment.json...")
        with open(f'{self.experiment_output_path}experiment.json', 'w') as file:
            json.dump(experiment, file, cls=CustomEncoder)

        self.logger.debug('Finished')
        return