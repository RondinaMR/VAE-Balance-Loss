import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    KBinsDiscretizer,
    PowerTransformer,
    QuantileTransformer,
)


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_bins: int = 0, transformer_type="Quantile", na_fill_value=None
    ) -> None:
        self.n_bins = n_bins
        self.transformer_type = transformer_type
        self.na_fill_value = na_fill_value
        self.scaler = None
        self.min = None
        self.max = None

    def fit(self, X, y=None):
        data = X.copy()
        # data = data.sample(n=min(data.shape[0], int(1e4)))
        if self.na_fill_value is None:
            strategy = "most_frequent"
        else:
            strategy = "constant"

        self.min = np.nanmin(data, axis=0)
        self.max = np.nanmax(data, axis=0)
#        print('nuovo_preprocessor3')
#        data = data + data * 0.1 * np.random.randn(data.shape[0], data.shape[1])
        data = (data - self.min) / (self.max - self.min)

        #data = np.clip(data, self.min*0, self.max*0+1.)
        self.imputer = SimpleImputer(
            strategy=strategy, add_indicator=False, fill_value=self.na_fill_value
        )
        self.imputer.fit(data)
        data = self.imputer.transform(data)

        if self.n_bins > 0:
            self.est = KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="kmeans"
            )
            self.est.fit(data)
            data = self.est.transform(data)
        else:
            if self.transformer_type == "Power":
                self.scaler = PowerTransformer()
            elif self.transformer_type == "Quantile":
                self.scaler = QuantileTransformer(
                    output_distribution="normal", random_state=0
                )
            else:
                self.scaler = MinMaxScaler()

            self.scaler.fit(data)
            data = self.scaler.transform(data)

        return self

    def get_feature_names(self):
        return []

    def transform(self, X, y=None):
        X = X.copy()

        X = (X - self.min) / (self.max - self.min)

        X = self.imputer.transform(X)
        if self.n_bins > 0:
            X = self.est.transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def inverse_transform(self, X, y=None):
        X = X.copy()

        if self.n_bins > 0:
            X = self.est.inverse_transform(X)

        else:
            X = self.scaler.inverse_transform(X)

        X[X <= self.na_fill_value] = np.nan

        X = self.min + (self.max - self.min) * X

        return X
