import numpy as np
import pandas as pd

from mlcf.error_injectors.abstract_error_injector import AbstractErrorInjector


class RandomNullsInjectorV2(AbstractErrorInjector):
    """
    Nulls Injector takes three arguments as input: a list of column names to affect, the percentage of rows to
     inject nulls, and the maximum number of nulls in one row. Then, it randomly selects indexes of rows
      to affect and randomly chooses columns where to place nulls for each selected row. The number of nulls in
       one selected row is a random number from 1 to the input maximum number of nulls in one row. In such a way,
        users have control under both rows and columns and can simulate real-world scenarios when there are multiple
        random nulls in one selected row.

    Parameters
    ----------
    seed
        Seed for all randomized operations in this generator
    columns_nulls_percentage_dct
        Dictionary where keys are column names and values are target percentages of nulls for a column

    """
    def __init__(self, seed: int, columns_to_transform: list, row_idx_nulls_percentage: float):
        super().__init__(seed)
        self.columns_to_transform = columns_to_transform
        self.row_idx_nulls_percentage = row_idx_nulls_percentage

    def _validate_input(self, df):
        for col in self.columns_to_transform:
            if col not in df.columns:
                raise ValueError(f"Value caused the issue is {col}. "
                                 f"Keys in columns_to_transform must be the dataframe column names")

        if self.row_idx_nulls_percentage < 0 or self.row_idx_nulls_percentage > 1:
            raise ValueError("Column nulls percentage must be in [0.0-1.0] range.")

    def set_percentage_var(self, new_row_idx_nulls_percentage):
        self.row_idx_nulls_percentage = new_row_idx_nulls_percentage

    def fit(self, df, target_column: str = None):
        self._validate_input(df)

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy(deep=True)
        if self.row_idx_nulls_percentage == 0.0:
            return df_copy

        nulls_sample_size = int(df_copy.shape[0] * self.row_idx_nulls_percentage)
        np.random.seed(self.seed)
        random_row_idxs = np.random.choice(df_copy.index, size=nulls_sample_size, replace=False)

        np.random.seed(self.seed)
        random_columns = np.random.choice(self.columns_to_transform, size=nulls_sample_size, replace=True)

        random_sample_df = pd.DataFrame({'column': random_columns, 'random_idx': random_row_idxs})
        for idx, col_name in enumerate(self.columns_to_transform):
            col_random_row_idxs = random_sample_df[random_sample_df['column'] == col_name]['random_idx'].values
            if col_random_row_idxs.shape[0] == 0:
                continue

            df_copy.loc[col_random_row_idxs, col_name] = None

        return df_copy

    def fit_transform(self, df, target_column: str = None):
        self.fit(df, target_column)
        transformed_df = self.transform(df)
        return transformed_df
