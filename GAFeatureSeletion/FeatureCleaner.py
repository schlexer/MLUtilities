# Data handling helper functions

import pandas as pd

# Pandas output
pd.set_option('display.max_columns', 30)


def delete_invariant_features(df, verbose=False):
    """Deletes columns that have only a single entry (value)."""
    drop_columns = []
    for column in df.columns:
        if len(df[column].unique()) == 1:
            if verbose:
                print(f'Single entry in: {column}')
            drop_columns.append(column)

    features = list(set(df.columns) - set(drop_columns))
    df = df[features]
    return df


def get_invariant_features(df, verbose=False):
    """Deletes columns that have only a single entry (value)."""
    drop_columns = []
    for column in df.columns:
        if len(df[column].unique()) == 1:
            if verbose:
                print(f'Single entry in: {column}')
            drop_columns.append(column)

    return drop_columns


def remove_outliers(df, column, center='median', std=3, max_value=None):
    if max_value is not None:
        df = df[df[column] < max_value]

    else:
        if not center in ['median', 'mean', 'avg']:
            print('Center options are median or mean.')
            return None

        if center == 'median':
            mu = df[column].median()
        else:
            mu = df[column].mean()

        std_sample = df[column].std()

        upper = mu + std*std_sample
        lower = mu - std*std_sample

        df = df[df[column].between(lower, upper)]

    return df


