from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import logging

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.logger = logging.getLogger('engine.transformers.CategoricalTransformer')
        pass

    def fit(self, X, y=None):
        self.X = X
        data = X.copy().astype(str)
        # data = data.sample(n=min(data.shape[0], int(1e4)))
        # self.logger.info(f"fitting categorical transformer")
        # self.logger.info(f"data type: {type(data)} - data shape: {data.shape}")
        # self.logger.info(data)
        # self.logger.info(f"number of class for each column: {data.nunique()}")
        self.imputer = SimpleImputer(strategy="most_frequent", add_indicator=False)
        self.imputer.fit(data)
        data = self.imputer.transform(data)
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(data)
        # self.logger.info(f"fitting done")
        # self.logger.info(data)
        return self

    def get_feature_names(self):
        return self.encoder.get_feature_names_out()

    def transform(self, X, y=None):
        X = X.copy().astype(str)
        X = self.imputer.transform(X)
        X = self.encoder.transform(X)
        return X

    def inverse_transform(self, X, y=None):
        X = X.copy()
        X = self.encoder.inverse_transform(X)
        return X
