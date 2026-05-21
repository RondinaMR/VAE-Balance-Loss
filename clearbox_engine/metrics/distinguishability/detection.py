import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score

from cbx_engine import Dataset, Preprocessor


class DetectionScore:
    original_dataset: Dataset
    synthetic_dataset: Dataset
    preprocessor: Preprocessor

    def __init__(
        self,
        original_dataset: Dataset,
        synthetic_dataset: Dataset,
        preprocessor: Preprocessor = None,
    ):
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.preprocessor = (
            preprocessor if preprocessor is not None else Preprocessor(original_dataset)
        )

    def get(self, features_to_hide: list = []):
        """
        Merges the original dataset with the newly generated synthetic one, using the result to feed an XGB Model. Such model
        is trained to predict whether an instance is synthetic or not. The lower the accuracy of this model, the higher the
        quality of the generated synthetic dataset.
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        original_df = self.original_dataset.data.sample(
            n=len(self.synthetic_dataset.data)
        ).copy()

        for i in self.preprocessor.discarded[1].keys():
            list_minority_labels = self.preprocessor.discarded[1][i]
            for j in list_minority_labels:
                original_df[i] = original_df[i].replace(j, '*')

        preprocessed_original_df = self.preprocessor.transform(original_df)
        original_df = self.preprocessor.reverse_transform(preprocessed_original_df)
        original_df["label"] = np.zeros(len(original_df)).astype(int)

        synthetic_df = self.synthetic_dataset.data.copy()
        preprocessed_synthetic_df = self.preprocessor.transform(synthetic_df)
        synthetic_df = self.preprocessor.reverse_transform(preprocessed_synthetic_df)
        synthetic_df["label"] = np.ones(len(synthetic_df)).astype(int)

        column_types = (
            self.original_dataset.column_types.copy()
            if self.original_dataset.column_types
            else None
        )
        if column_types:
            cols = [col for col in column_types]
            for col in cols:
                if col not in original_df.columns.tolist():
                    column_types.pop(col)
            column_types["label"] = "number"

        dataset = Dataset(
            data=pd.concat([original_df, synthetic_df])
            .sample(frac=1)
            .reset_index(drop=True),
            target_column="label",
            regression=False,
            column_types=column_types,
        )

        preprocessor = Preprocessor(dataset)

        # Starting with cbx AI
        Y = dataset.get_y().values
        
        # split data into train and test sets
        seed = 42
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.get_x(), Y, test_size=test_size, random_state=seed
        )

        train_ds = preprocessor.transform(X_train)
        test_ds = preprocessor.transform(X_test)

        model = xgb.XGBClassifier(max_depth=3, n_estimators=50, use_label_encoder=False, eval_metric="logloss")
        model.fit(train_ds, y_train)

        y_pred = model.predict(test_ds)

        detection_score = {}
        detection_score["accuracy"] = round(
            accuracy_score(y_true=y_test, y_pred=y_pred), 4
        )
        detection_score["ROC_AUC"] = round(
            roc_auc_score(
                y_true=y_test,
                y_score=model.predict_proba(test_ds)[:, 1],
                average=None,
            ),
            4,
        )
        detection_score["score"] = (
            1
            if detection_score["ROC_AUC"] <= 0.5
            else (1 - detection_score["ROC_AUC"]) * 2
        )

        detection_score["feature_importances"] = {}

        (
            ordinal_features_sizes,
            categorical_features_sizes,
        ) = preprocessor.get_features_sizes()
        preprocessed_ordinal_features = []
        preprocessed_categorical_features = []
        preprocessed_datetime_features = []

        if ordinal_features_sizes:
            preprocessed_ordinal_features = preprocessor.transformer.transformers[0][2]
            if preprocessor.get_datetime_features():
                preprocessed_datetime_features = preprocessor.transformer.transformers[
                    1
                ][2]
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        preprocessor.transformer.transformers[2][2]
                    )
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        preprocessor.transformer.transformers[1][2]
                    )
        else:
            if preprocessor.get_datetime_features():
                preprocessed_datetime_features = preprocessor.transformer.transformers[
                    0
                ][2]
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        preprocessor.transformer.transformers[1][2]
                    )
            else:
                if categorical_features_sizes:
                    preprocessed_categorical_features = (
                        preprocessor.transformer.transformers[0][2]
                    )

        index = 0

        for feature, importance in zip(
            preprocessed_ordinal_features, model.feature_importances_
        ):
            if feature not in features_to_hide:
                detection_score["feature_importances"][feature] = round(
                    float(importance), 4
                )
            index += 1

        if preprocessed_datetime_features:
            for feature, importance in zip(
                preprocessed_datetime_features, model.feature_importances_[index:]
            ):
                if feature not in features_to_hide:
                    detection_score["feature_importances"][feature] = round(
                        float(importance), 4
                    )
                index += 1

        for feature, feature_size in zip(
            preprocessed_categorical_features, categorical_features_sizes
        ):
            importance = np.sum(
                model.feature_importances_[index : index + feature_size]
            )
            index += feature_size
            if feature not in features_to_hide:
                detection_score["feature_importances"][feature] = round(
                    float(importance), 4
                )

        return detection_score
