import json

import pandas as pd
import numpy as np

from cbx_engine import Dataset, Preprocessor


def _autocorr(x: pd.Series):
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2 :]


class Autocorrelation:
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

    def get(self, feature: str, id: str = None):
        original_data = self.original_dataset.data.copy()
        if self.original_dataset.sequence_index:
            original_data = original_data.set_index(self.original_dataset.sequence_index)
        if id:
            original_data = original_data.loc[
                original_data[self.original_dataset.group_by] == id
            ]

        original_x = np.array(original_data[feature])
        original_z = _autocorr(original_x)
        original_z = original_z / float(original_z.max())
        original_area = round(float(np.trapz(original_z)), 4)

        synthetic_data = self.synthetic_dataset.data.copy()
        if self.original_dataset.sequence_index:
            synthetic_data = synthetic_data.set_index(self.original_dataset.sequence_index)
        if id and self.original_dataset.group_by:
            synthetic_data = synthetic_data.loc[
                synthetic_data[self.original_dataset.group_by] == id
            ]
        synthetic_x = np.array(synthetic_data[feature])
        synthetic_z = _autocorr(synthetic_x)
        synthetic_z = synthetic_z / float(synthetic_z.max())
        synthetic_area = round(float(np.trapz(synthetic_z)), 4)

        autocorrelation = {}
        autocorrelation["original"] = json.dumps(original_z.tolist())
        autocorrelation["original_area"] = original_area
        autocorrelation["synthetic"] = json.dumps(synthetic_z.tolist())
        autocorrelation["synthetic_area"] = synthetic_area
        autocorrelation["diff_area"] = round(
            float(abs(original_area - synthetic_area)), 4
        )

        return autocorrelation
