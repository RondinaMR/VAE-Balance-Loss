import random

from cbx_engine import Dataset, Preprocessor


class QueryPower:
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

    def get(self):
        """
        Merges the original dataset with the newly generated synthetic one, using the result to feed an XGB Model. Such model
        is trained to predict whether an instance is synthetic or not. The lower the accuracy of this model, the higher the
        quality of the generated synthetic dataset.
        """
        query_power = {}
        query_power["queries"] = []

        original_df = self.original_dataset.data.sample(
            n=len(self.synthetic_dataset.data)
        ).copy()
        preprocessed_original_df = self.preprocessor.transform(original_df)
        original_df = self.preprocessor.reverse_transform(preprocessed_original_df)

        synthetic_df = self.synthetic_dataset.data.copy()
        preprocessed_synthetic_df = self.preprocessor.transform(synthetic_df)
        synthetic_df = self.preprocessor.reverse_transform(preprocessed_synthetic_df)

        ordinal_features = self.preprocessor.get_ordinal_features()
        categorical_features = self.preprocessor.get_categorical_features()
        datetime_features = self.preprocessor.get_datetime_features()

        features = original_df.columns.tolist()
        features = list(set(features) - set(datetime_features))

        quantiles = [0.25, 0.5, 0.75]
        numerical_ops = ["<=", ">="]
        categorical_ops = ["==", "!="]
        logical_ops = ["and"]

        queries_score = []

        while len(features) >= 2 and len(query_power["queries"]) < 5:
            feats = []
            feats.append(random.choice(features))
            features.remove(feats[0])

            feats.append(random.choice(features))
            features.remove(feats[1])

            queries = []

            for feature in feats:
                if feature in ordinal_features:
                    op = random.choice(numerical_ops)
                    value = original_df.quantile(q=random.choice(quantiles), numeric_only=True)[feature]
                elif feature in categorical_features:
                    op = random.choice(categorical_ops)
                    value = random.choice(original_df[feature].unique())
                    value = f"'{value}'"

                queries.append(f"`{feature}` {op} {value}")

            text = f" {random.choice(logical_ops)} ".join(queries)
            try:
                query = {
                    "text": text,
                    "original_df": len(original_df.query(text)),
                    "synthetic_df": len(synthetic_df.query(text)),
                }
            except:
                query = {
                    "text": 'Invalid query',
                    "original_df": 0,
                    "synthetic_df": 0,
                }

            query_power["queries"].append(query)
            queries_score.append(
                1 - abs(query["original_df"] - query["synthetic_df"]) / len(original_df)
            )

        query_power["score"] = round(float(sum(queries_score) / len(queries_score)), 4)

        return query_power
