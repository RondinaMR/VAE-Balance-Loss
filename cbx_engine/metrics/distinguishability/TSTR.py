import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
)

from cbx_engine import Dataset, Preprocessor


class TSTRScore:
    original_dataset: Dataset
    synthetic_dataset: Dataset
    preprocessor: Preprocessor

    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        validation_dataset: Dataset,
        preprocessor: Preprocessor = None,
    ):
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.validation_dataset = validation_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self, features_to_hide: list = []):
        """
        Two XGB models are trained: one using the original training dataset and one using the newly synthetic datasets. Both
        models are then tested on the original validation dataset, to check wether accuracy (or MSE for regression task)
        changes or not.
        """
        n_rows = min(
            self.original_dataset.get_x().shape[0],
            self.synthetic_dataset.get_x().shape[0],
        )

        preprocessed_original_dataset = self.preprocessor.transform(
            self.original_dataset.get_x().head(n_rows)
        )
        preprocessed_synthetic_dataset = self.preprocessor.transform(
            self.synthetic_dataset.get_x().head(n_rows)
        )
        preprocessed_validation_dataset = self.preprocessor.transform(
            self.validation_dataset.get_x()
        )

        TSTR_score = {}
        TSTR_score["feature_importances"] = {}
        TSTR_score["feature_importances"]["training"] = {}
        TSTR_score["feature_importances"]["synthetic"] = {}

        if self.original_dataset.regression:
            from xgboost import XGBRegressor as xgb

            TSTR_score["task"] = "regression"
            TSTR_score["MSE"] = {}
            TSTR_score["RMSE"] = {}
            TSTR_score["MAE"] = {}
            TSTR_score["max_error"] = {}
            TSTR_score["r2_score"] = {}

            training_Y = self.original_dataset.get_y().values[:n_rows]
            
            synthetic_Y = self.synthetic_dataset.get_y().values[:n_rows]

            model_original_data = xgb()
            model_original_data.fit(preprocessed_original_dataset, training_Y)

            predictions = model_original_data.predict(preprocessed_validation_dataset)
            TSTR_score["MSE"]["training"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            TSTR_score["RMSE"]["training"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                        squared=False,
                    ),
                    4,
                )
            )
            TSTR_score["MAE"]["training"] = float(
                round(
                    mean_absolute_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            TSTR_score["max_error"]["training"] = float(
                round(
                    max_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            TSTR_score["r2_score"]["training"] = float(
                round(
                    r2_score(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )

            model_synthetic_data = xgb()
            model_synthetic_data.fit(preprocessed_synthetic_dataset, synthetic_Y)

            predictions = model_synthetic_data.predict(preprocessed_validation_dataset)
            TSTR_score["MSE"]["synthetic"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            TSTR_score["RMSE"]["synthetic"] = float(
                round(
                    mean_squared_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                        squared=False,
                    ),
                    4,
                )
            )
            TSTR_score["MAE"]["synthetic"] = float(
                round(
                    mean_absolute_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            TSTR_score["max_error"]["synthetic"] = float(
                round(
                    max_error(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            TSTR_score["r2_score"]["synthetic"] = float(
                round(
                    r2_score(
                        y_true=self.validation_dataset.get_y().values,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            TSTR_score["score"] = round(
                float(
                    1
                    - (
                        abs(
                            TSTR_score["MAE"]["training"]
                            - TSTR_score["MAE"]["synthetic"]
                        )
                        / max(
                            TSTR_score["max_error"]["training"],
                            TSTR_score["max_error"]["synthetic"],
                        )
                    )
                ),
                4,
            )
        else:
            from xgboost import XGBClassifier as xgb

            Y, Y_labels = self.validation_dataset.get_label_encoded_y()

            TSTR_score["task"] = "classification"
            TSTR_score["accuracy"] = {}
            TSTR_score["metrics"] = {}
            TSTR_score["metrics"]["training"] = []
            TSTR_score["metrics"]["synthetic"] = []
            training_Y = self.original_dataset.get_y()
            training_Y = training_Y.astype('category')
            training_Y = training_Y.cat.codes           
            training_Y = training_Y.values[:n_rows]
                        
            synthetic_Y = self.synthetic_dataset.get_y()[:n_rows]
            synthetic_Y = synthetic_Y.astype('category')
            synthetic_Y = synthetic_Y.cat.codes           
            synthetic_Y = synthetic_Y.values[:n_rows]            


            y_true = self.validation_dataset.get_y()
            y_true = y_true.astype('category')
            y_true = y_true.cat.codes           
            y_true = y_true.values[:n_rows]                 

            model_original_data = xgb(eval_metric="logloss")

            model_original_data.fit(preprocessed_original_dataset, training_Y)

            predictions = model_original_data.predict(preprocessed_validation_dataset)
            TSTR_score["accuracy"]["training"] = float(
                round(
                    accuracy_score(
                        y_true=y_true,
                        y_pred=predictions,
                    ),
                    4,
                )
            )
            precisions, recalls, fscores, supports = precision_recall_fscore_support(
                y_true, predictions
            )
            for label in np.unique(Y):
                TSTR_score["metrics"]["training"].append(
                    {
                        "label": str(Y_labels[label]),
                        "precision": float(round(precisions[label], 4)),
                        "recall": float(round(recalls[label], 4)),
                        "fscore": float(round(fscores[label], 4)),
                        "support": float(round(supports[label], 4)),
                    }
                )

            model_synthetic_data = xgb()
            model_synthetic_data.fit(preprocessed_synthetic_dataset, synthetic_Y)

            predictions = model_synthetic_data.predict(preprocessed_validation_dataset)
            TSTR_score["accuracy"]["synthetic"] = round(
                accuracy_score(
                    y_true=y_true,
                    y_pred=predictions,
                ),
                4,
            )
            precisions, recalls, fscores, supports = precision_recall_fscore_support(
                y_true, predictions
            )
            for label in np.unique(Y):
                TSTR_score["metrics"]["synthetic"].append(
                    {
                        "label": str(Y_labels[label]),
                        "precision": float(round(precisions[label], 4)),
                        "recall": float(round(recalls[label], 4)),
                        "fscore": float(round(fscores[label], 4)),
                        "support": float(round(supports[label], 4)),
                    }
                )

            TSTR_score["score"] = round(
                float(
                    1
                    - (
                        abs(
                            TSTR_score["accuracy"]["training"]
                            - TSTR_score["accuracy"]["synthetic"]
                        )
                    )
                ),
                4,
            )

        (
            ordinal_features_sizes,
            categorical_features_sizes,
        ) = self.preprocessor.get_features_sizes()
        preprocessed_ordinal_features = []
        preprocessed_categorical_features = []
        preprocessed_datetime_features = []

        if ordinal_features_sizes:
            preprocessed_ordinal_features = self.preprocessor.transformer.transformers[
                0
            ][2]
            if self.preprocessor.get_datetime_features():
                preprocessed_datetime_features = (
                    self.preprocessor.transformer.transformers[1][2]
                )
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[2][2]
                    )
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[1][2]
                    )
        else:
            if self.preprocessor.get_datetime_features():
                preprocessed_datetime_features = (
                    self.preprocessor.transformer.transformers[0][2]
                )
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[1][2]
                    )
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        self.preprocessor.transformer.transformers[0][2]
                    )

        index = 0

        for feature, importance in zip(
            preprocessed_ordinal_features, model_original_data.feature_importances_
        ):
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["training"][feature] = round(
                    float(importance), 4
                )
            index += 1

        if preprocessed_datetime_features:
            for feature, importance in zip(
                preprocessed_datetime_features,
                model_original_data.feature_importances_[index:],
            ):
                if feature not in features_to_hide:
                    TSTR_score["feature_importances"]["training"][feature] = round(
                        float(importance), 4
                    )
                index += 1

        for feature, feature_size in zip(
            preprocessed_categorical_features, categorical_features_sizes
        ):
            importance = np.sum(
                model_original_data.feature_importances_[index : index + feature_size]
            )
            index += feature_size
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["training"][feature] = round(
                    float(importance), 4
                )

        index = 0

        for feature, importance in zip(
            preprocessed_ordinal_features, model_synthetic_data.feature_importances_
        ):
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["synthetic"][feature] = round(
                    float(importance), 4
                )
            index += 1

        if preprocessed_datetime_features:
            for feature, importance in zip(
                preprocessed_datetime_features,
                model_synthetic_data.feature_importances_[index:],
            ):
                if feature not in features_to_hide:
                    TSTR_score["feature_importances"]["synthetic"][feature] = round(
                        float(importance), 4
                    )
                index += 1

        for feature, feature_size in zip(
            preprocessed_categorical_features, categorical_features_sizes
        ):
            importance = np.sum(
                model_synthetic_data.feature_importances_[index : index + feature_size]
            )
            index += feature_size
            if feature not in features_to_hide:
                TSTR_score["feature_importances"]["synthetic"][feature] = round(
                    float(importance), 4
                )

        return TSTR_score
