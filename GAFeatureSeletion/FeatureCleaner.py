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


def remove_outliers(df,
                    column,
                    percentile=0.01,
                    stdev=None,
                    center="mean",
                    max_value=None):

    if max_value is not None:
        df = df[df[column] < max_value]

    if center not in ['median', 'mean']:
        print('Center options are median or mean.')
        return None

    if percentile is not None:
        print("Using percentile.")
        lower = df[column].quantile(percentile)
        upper = df[column].quantile(1 - percentile)

    if stdev is not None:
        if center == "mean":
            mu = df[column].mean()
        else:
            mu = df[column].median()

        std_sample = df[column].std()
        upper = mu + stdev * std_sample
        lower = mu - stdev * std_sample

    df = df[df[column].between(lower, upper)]

    return df


