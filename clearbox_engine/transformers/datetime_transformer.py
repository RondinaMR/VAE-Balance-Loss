import dateinfer
import numpy as np

from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.datetime_formats = []
        self.dividers = []

    def _find_datetime_format(self, data):
        try:
            for datetime_column in data:
                self.datetime_formats.append(
                    dateinfer.infer(
                        data[datetime_column].head(min(500, len(data))).astype(str)
                    )
                )
        except Exception:
            self.datetime_formats.append("N")

    def _find_divider(self, data):
        # divider = 1
        # multipliers = [10] * 9 + [60, 60, 24]
        # for multiplier in multipliers:
        #     candidate = divider * multiplier
        #     if (data % candidate).any():
        #         break

        #     divider = candidate
        # return divider
        if (data > 1e9).any():
            return 1e9
        else:
            return 1

    def fit(self, X, y=None):
        self._find_datetime_format(X)

        data = X.copy().astype("datetime64", errors="ignore").astype("int")
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.fit(data)
        data = self.imputer.transform(data.astype("int", errors="ignore"))
        for i in range(data.shape[-1]):
            divider = self._find_divider(data[:, i])
            self.dividers.append(divider)
            data[:, i] = data[:, i] // divider

        return self

    def get_feature_names(self):
        return []

    def transform(self, X, y=None):
        X = X.copy().astype("datetime64", errors="ignore").astype("int")
        X = self.imputer.transform(X)
        for i in range(X.shape[-1]):
            X[:, i] = X[:, i] // self.dividers[i]

        return X

    def inverse_transform(self, X, y=None):
        X = X.copy()

        datetimes = np.empty(shape=X.shape, dtype="object")

        for i in range(X.shape[-1]):
            for j, date in enumerate(X[:, i]):
                datetimes[:, i][j] = datetime.fromtimestamp(date).strftime(
                    self.datetime_formats[i]
                )

        return datetimes
