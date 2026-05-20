import sklearn.pipeline
import sklearn.compose
import sklearn.preprocessing
import sklearn.impute
import pandas as pd
import numpy as np

from typing import Dict, List, Callable, Tuple
from loguru import logger
import logging
# create logger
module_logger = logging.getLogger('engine.preprocessor')

from clearbox_engine import (
    Dataset,
    OrdinalTransformer,
    CategoricalTransformer,
    DatetimeTransformer,
)


class Preprocessor:
    transformer: sklearn.compose.ColumnTransformer
    not_fitted_transformer: sklearn.compose.ColumnTransformer
    inverse_transformer: Callable
    discarded: Tuple
    sorted_columns = List
    ordinal_features = List
    categorical_features = List
    datetime_features = List

    def __init__(
        self,
        dataset: Dataset,
        threshold: float = -1.0,  # was 0.02 (deleted to avoid deleting labels with frequency < 0.02)
        n_ordinal_bins: int = 0,
        num_transformer_type: str = "Quantile",
        na_fill_value: float = -0.001,
    ):
        self.logger = logging.getLogger('engine.preprocessor.Preprocessor')
        X = dataset.get_x().copy()#.sample(n=min(dataset.get_x().shape[0], int(1e4))).copy()
        self.logger.debug("Preprocessor init")
        # self.logger.debug(f"Number of classes in 'work_class': {X['work_class'].nunique()}")
        # self.logger.debug(str(X["work_class"].value_counts()))
        # self.logger.debug(f"Number of classes in 'occupation': {X['occupation'].nunique()}")
        # self.logger.debug(str(X["occupation"].value_counts()))
        (
            self.ordinal_features,
            self.categorical_features,
            self.datetime_features,
        ) = self._infer_feature_types(dataset)
        self.logger.debug("infer feature types done")
        self.logger.debug(f"self.categorical_features: {self.categorical_features}")
        # self.logger.debug(f"Number of classes in 'work_class': {X['work_class'].nunique()}")
        # self.logger.debug(f"Number of classes in 'occupation': {X['occupation'].nunique()}")
        X = self._feature_selection(
            X, self.categorical_features, self.ordinal_features, threshold
        )
        self.logger.debug("feature selection done")
        # self.logger.debug(f"Number of classes in 'work_class': {X['work_class'].nunique()}")
        # self.logger.debug(f"Number of classes in 'occupation': {X['occupation'].nunique()}")
        X = X.sample(n=min(X.shape[0], int(1e4)))
        discarded = [discarded for discarded in self.discarded[0]]
        self.logger.debug(f"discarded columns: {discarded}")
        self.sorted_columns = [col for col in X.columns.values if col not in discarded]
        self.logger.debug(f"sorted columns: {discarded}")
        self.logger.debug("sample done")
        # self.logger.debug(f"Number of classes in 'work_class': {X['work_class'].nunique()}")
        # self.logger.debug(f"Number of classes in 'occupation': {X['occupation'].nunique()}")

        self.ordinal_features = [
            i
            for i in self.ordinal_features
            if i not in [j for j in self.discarded[0]]
        ]

        self.categorical_features = [
            i
            for i in self.categorical_features
            if i not in [j for j in self.discarded[0]]
        ]
        self.logger.debug(f"self.categorical_features: {self.categorical_features}")

        transformers_list = list()
        if len(self.ordinal_features) > 0:
            transformers_list.append(
                (
                    "ordinal_transformer",
                    OrdinalTransformer(
                        n_ordinal_bins, num_transformer_type, na_fill_value
                    ),
                    self.ordinal_features,
                )
            )
        if len(self.datetime_features) > 0:
            transformers_list.append(
                (
                    "datetime_transformer",
                    DatetimeTransformer(),
                    self.datetime_features,
                )
            )
        if len(self.categorical_features) > 0:
            transformers_list.append(
                (
                    "categorical_transformer",
                    CategoricalTransformer(),
                    self.categorical_features,
                )
            )

        column_transformer = sklearn.compose.ColumnTransformer(
            transformers=transformers_list
        )
        self.transformer = column_transformer.fit(X)
        
        self.not_fitted_transformer = column_transformer
        self.logger.debug("ColumnTransformer done")
        self.logger.debug(f"self.transformer: {self.transformer}")
        self.logger.debug(f"self.not_fitted_transformer: {self.not_fitted_transformer}")
        # self.logger.debug(f"Number of classes in 'work_class': {X['work_class'].nunique()}")
        # self.logger.debug(f"Number of classes in 'occupation': {X['occupation'].nunique()}")
        
        self.inverse_transformer = self._setup_inverse_preprocessor()

        self.logger.debug("setup inverse preprocessor done")
        # self.logger.debug(f"Number of classes in 'work_class': {X['work_class'].nunique()}")
        # self.logger.debug(f"Number of classes in 'occupation': {X['occupation'].nunique()}")

    @staticmethod
    def _infer_feature_types(dataset: Dataset) -> Tuple:
        """
        Use column types from dataset objects if defined, otherwise it infers dtypes from Pandas DataFrame.
        Returns a tuple of lists of column names for each type.
        """
        if dataset.column_types:
            ordinal_features = [
                column
                for column in dataset.column_types.keys()
                if (
                    dataset.column_types[column] == "number"
                    or dataset.column_types[column] == "boolean"
                )
                and column != dataset.target_column
                and column != dataset.sequence_index
                and column != dataset.group_by
            ]
            categorical_features = [
                column
                for column in dataset.column_types.keys()
                if dataset.column_types[column] == "string"
                and column != dataset.target_column
                and column != dataset.sequence_index
                and column != dataset.group_by
            ]
            datetime_features = [
                column
                for column in dataset.column_types.keys()
                if dataset.column_types[column] == "datetime"
                and column != dataset.target_column
                and column != dataset.sequence_index
                and column != dataset.group_by
            ]
        else:
            bool_features = dataset.x_columns(include=["bool"])
            dataset.data[bool_features] = dataset.data[bool_features].astype("category")

            datetime_features = dataset.x_columns(include=["datetime", "timedelta"])
            dataset.data[datetime_features] = dataset.data[datetime_features].astype(
                "int64"
            )
            datetime_features = []

            ordinal_features = dataset.x_columns(include=["number", "datetime"])
            categorical_features = dataset.x_columns(include=["object", "category"])

        return ordinal_features, categorical_features, datetime_features

    @staticmethod
    def _shrink_labels(instance, too_much_info: dict):
        for column_name in too_much_info:
            if instance[column_name].dtype == "object":
                # instance[column_name].replace(too_much_info[column_name], "*", inplace=True)
                instance[column_name] = instance[column_name].apply(
                    lambda x: x if x not in too_much_info[column_name] else "*"
                )
            else:
                # instance[column_name].replace(too_much_info[column_name], -999999, inplace=True)
                instance[column_name] = instance[column_name].apply(
                    lambda x: x if x not in too_much_info[column_name] else -999999
                )
        return instance

    def _feature_selection(
        self,
        X: pd.DataFrame,
        categorical_features: List[str],
        ordinal_features: List[str],
        threshold: float = -1.0,
    ) -> pd.DataFrame:
        """
        Perform a selection of the most useful columns for a given DataFrame, ignorig the other features.
        """
        cat_features_stats = [
            (
                i,
                X[i].value_counts(),
                X[i].nunique(),
                X.columns.get_loc(i),
            )
            for i in categorical_features
        ]

        ord_features_stats = [
            (
                i,
                X[i].value_counts(),
                X[i].unique(),
                X.columns.get_loc(i),
            )
            for i in ordinal_features
        ]

        no_info = []
        too_much_info = {}
        for column_stats in cat_features_stats:
            # introduced a cap on maximum number of unique labels
            if (column_stats[1].shape[0] == 1) or (
                column_stats[1].shape[0] >= (X.shape[0] * 0.98)
            ):
                no_info.append(column_stats[0])
            else:
                counts = column_stats[1].values / column_stats[1].values.sum()
                values_to_shrink_indices = np.where(counts < threshold)[0]
                if (
                    values_to_shrink_indices.shape[0] > 0
                    and column_stats[1].shape[0] > 2
                ):
                    too_much_info[column_stats[0]] = (
                        column_stats[1].index[values_to_shrink_indices].to_list()
                    )
                    # too_much_info.append(
                    #     (column_stats[0], column_stats[1].index[column_name])
                    # )

        for column_stats in ord_features_stats:
            if column_stats[1].shape[0] <= 1:
                no_info.append(column_stats[0])
        X = self._shrink_labels(X, too_much_info)
        self.discarded = (no_info, too_much_info)

        return X

    def _setup_inverse_preprocessor(self):
        preprocessor_input_columns: List = list()
        transformer_input_columns: Dict = dict()
        preprocessor_output_columns: List = list()
        inverse_preprocessors_map: Dict = dict()

        for w, transformer in enumerate(self.transformer.transformers_):
            if transformer[0] != "remainder":
                preprocessor_input_columns += transformer[-1]
                inv_transform = transformer[1].inverse_transform
                feat_names = transformer[1].get_feature_names()
                partial_output_columns = (
                    feat_names.tolist() if len(feat_names) > 0 else transformer[-1]
                )
                preprocessor_output_columns += partial_output_columns
                transformer_input_columns[w] = transformer[-1]
                inverse_preprocessors_map[
                    tuple(
                        [
                            preprocessor_output_columns.index(i)
                            for i in partial_output_columns
                        ]
                    )
                ] = inv_transform

        def inverse_preprocessor(encoded_matrix: np.ndarray) -> pd.DataFrame:
            """
            Return the non encoded version of a matrix of instances.
            Parameters
            ----------
            encoded_matrix : numpy array
                A matrix of encoded/pre-processed instances.
            Returns
            -------
            pd.Dataframe
                The non encoded version of the matrix encoded_matrix.
            """
            non_encoded_dataframe = pd.DataFrame(columns=preprocessor_input_columns)
            for j, (
                encoded_columns_indices,
                inverse_transform,
            ) in enumerate(inverse_preprocessors_map.items()):

                encoded_values = encoded_matrix[:, list(encoded_columns_indices)]
                decoded_values = encoded_values  # .reshape(1, -1)

                decoded_values = inverse_transform(decoded_values)

                for i1, i2 in enumerate(transformer_input_columns[j]):
                    non_encoded_dataframe[i2] = decoded_values[:, i1]

            return non_encoded_dataframe[preprocessor_input_columns]

        return inverse_preprocessor

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X = self._shrink_labels(X, self.discarded[1])

        for transformer in self.transformer.transformers_:
            if "categorical_transformer" in transformer:
                categories = [cat for cat in transformer[2]]

                X[categories] = X[categories].astype(str)

        x_batch_preprocessed = self.transformer.transform(X)

        return x_batch_preprocessed

    def reverse_transform(self, x):
        x = self.inverse_transformer(x).fillna(0)

        return x[self.sorted_columns]

    def get_features_sizes(self) -> Tuple:
        self.logger.debug("get_features_sizes")
        ordinal_sizes = list()
        categorical_sizes = list()
        for transformer in self.transformer.transformers_:
            if "ordinal_transformer" in transformer:
                ordinal_sizes.append(len(transformer[-1]))
            if "datetime_transformer" in transformer:
                if ordinal_sizes:
                    ordinal_sizes[0] += len(transformer[-1])
                else:
                    ordinal_sizes.append(len(transformer[-1]))

            if "categorical_transformer" in transformer:
                self.logger.debug(f"transformer[1]: {transformer[1]}")
                one_hot_encoder = transformer[1].encoder
                self.logger.debug(f"one_hot_encoder: {one_hot_encoder}")
                self.logger.debug(f"one_hot_encoder.categories_:\n{one_hot_encoder.categories_}")
                categorical_sizes = [len(cat) for cat in one_hot_encoder.categories_]
                self.logger.debug(f"categorical_sizes: {categorical_sizes}")

        return ordinal_sizes, categorical_sizes

    def get_ordinal_features(self) -> List:
        return self.ordinal_features.copy()

    def get_categorical_features(self) -> List:
        return self.categorical_features.copy()

    def get_datetime_features(self) -> List:
        return self.datetime_features.copy()
