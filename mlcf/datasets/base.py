import pandas as pd


class BaseDataLoader:
    """
    Base data loader class that is used as input for the whole ML lifecycle.

    Parameters
    ----------
    full_df
        Full dataset in a pandas dataframe format
    target
        Name of the target column name
    numerical_columns
        List of numerical column names
    categorical_columns
        List of categorical column names
    columns_with_nulls
        [Optional] List of column names that contains nulls
    subsample_size
        [Optional] Size of a subsample to create based on the input dataset
    subsample_seed
        [Optional] Seed for dataset subsampling

    """
    def __init__(self, full_df: pd.DataFrame, target: str, numerical_columns: list, categorical_columns: list,
                 columns_with_nulls: list = None, subsample_size: int = None, subsample_seed: int = None):
        if subsample_size:
            df = full_df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else full_df.sample(subsample_size)
            df = df.reset_index(drop=True)
        else:
            df = full_df

        self.full_df = df
        self.target = target
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.features = numerical_columns + categorical_columns

        self.X_data = self.full_df[self.features]
        self.y_data = self.full_df[self.target]
        self.columns_with_nulls = self.X_data.columns[self.X_data.isna().any().to_list()].to_list() \
            if columns_with_nulls is None else columns_with_nulls
