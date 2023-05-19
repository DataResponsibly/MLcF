import pathlib
import pandas as pd

from mlcf.datasets.base import BaseDataLoader


class CreditDataset(BaseDataLoader):
    """
    Dataset class for Credit dataset that contains sensitive attributes among feature columns.
    Source: https://www.kaggle.com/competitions/GiveMeSomeCredit/overview

    Parameters
    ----------
    subsample_size
        Subsample size to create based on the input dataset
    subsample_seed
        Seed for sampling using pandas Dataframe sample() method

    """
    def __init__(self, subsample_size: int = None, subsample_seed: int = None):
        filename = 'givemesomecredit.csv'
        dataset_path = pathlib.Path(__file__).parent.joinpath(filename)

        df = pd.read_csv(dataset_path, index_col=0)
        if subsample_size:
            df = df.sample(subsample_size, random_state=subsample_seed) if subsample_seed is not None \
                else df.sample(subsample_size)
            df = df.reset_index(drop=True)

        target = 'SeriousDlqin2yrs'
        numerical_columns = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                             'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                             'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines']
        categorical_columns = []

        super().__init__(
            full_df=df,
            target=target,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )
