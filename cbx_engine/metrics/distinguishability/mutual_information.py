import pandas as pd

from sklearn.metrics.cluster import normalized_mutual_info_score

from cbx_engine import Dataset, Preprocessor


class MutualInformation:
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
            preprocessor if preprocessor is not None else Preprocessor(original_dataset, n_ordinal_bins=10)
        )

    def get(self, features_to_hide: list = []):
        original_df = \
            self.preprocessor.transform(self.original_dataset.get_x().sample(
                n=min(10000, self.original_dataset.data.shape[0])))
        original_df = self.preprocessor.reverse_transform(original_df)

        for i in original_df.columns:
            if original_df[i].dtype == 'object':
                original_df[i] = original_df[i].fillna("other")
            else:
                original_df[i] = pd.cut(original_df[i].fillna(0), 5)

        # for i in self.preprocessor.ordinal_features:
        #     original_df[i] = original_df[i].fillna(0)
        # for i in self.preprocessor.categorical_features:
        #     original_df[i] = original_df[i].fillna("other")

        synthetic_df = \
            self.preprocessor.transform(self.synthetic_dataset.get_x().sample(
                n=min(10000, self.synthetic_dataset.data.shape[0])))
        synthetic_df = self.preprocessor.reverse_transform(synthetic_df)

        for i in synthetic_df.columns:
            if synthetic_df[i].dtype == 'object':
                synthetic_df[i] = synthetic_df[i].fillna("other")
            else:
                synthetic_df[i] = pd.cut(synthetic_df[i].fillna(0), 5)


        # for i in self.preprocessor.ordinal_features:
        #     synthetic_df[i] = synthetic_df[i].fillna(0)
        #
        # for i in self.preprocessor.categorical_features:
        #     synthetic_df[i] = synthetic_df[i].fillna("other")

        mutual_information = {}

        mutual_information["features"] = [
            item
            for item in synthetic_df.columns
            if item not in self.preprocessor.get_datetime_features()
            and item not in features_to_hide
        ]

        original_mutual_information = [
            [0 for _ in range(len(mutual_information["features"]))]
            for _ in range(len(mutual_information["features"]))
        ]
        synthetic_mutual_information = [
            [0 for _ in range(len(mutual_information["features"]))]
            for _ in range(len(mutual_information["features"]))
        ]
        diff_correlation_matrix = [
            [0 for _ in range(len(mutual_information["features"]))]
            for _ in range(len(mutual_information["features"]))
        ]

        for i, feature_i in enumerate([col for col in mutual_information["features"]]):
            for j, feature_j in enumerate(
                [col for col in mutual_information["features"]]
            ):

                try:
                    result = float(
                        round(
                            normalized_mutual_info_score(
                                original_df[feature_i], original_df[feature_j]
                            ),
                            4,
                        )
                    )
                except:
                    result = float(0)

                original_mutual_information[i][j] = (
                    "NaN"
                    if pd.isnull(result)
                    else float(result)
                    if isinstance(result, float)
                    else int(result)
                )

                try:
                    result = float(
                        round(
                            normalized_mutual_info_score(
                                synthetic_df[feature_i], synthetic_df[feature_j]
                            ),
                            4,
                        )
                    )
                except:
                    result = float(0)

                synthetic_mutual_information[i][j] = (
                    "NaN"
                    if pd.isnull(result)
                    else float(result)
                    if isinstance(result, float)
                    else int(result)
                )

                diff_correlation_matrix[i][j] = (
                    "NaN"
                    if (
                        original_mutual_information[i][j] == "NaN"
                        or synthetic_mutual_information[i][j] == "NaN"
                    )
                    else float(
                        round(
                            abs(
                                original_mutual_information[i][j]
                                - synthetic_mutual_information[i][j]
                            ),
                            4,
                        )
                    )
                )

        mutual_information["original_mutual_information"] = original_mutual_information
        mutual_information[
            "synthetic_mutual_information"
        ] = synthetic_mutual_information
        mutual_information["diff_correlation_matrix"] = diff_correlation_matrix

        score = 0
        for row in mutual_information["diff_correlation_matrix"]:
            score += sum(row)
        mutual_information["score"] = round(
            float(1 - score / (len(mutual_information["features"]) ** 2)), 4
        )

        return mutual_information
