from configs import WORKING_DIR, INPUT_DIR
import os
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer, make_column_selector
from typing import Union

NUMERICS_LIST = ['float16', 'float32', 'float64']

def get_file_path(fname=None):
    input_dir = os.path.join(WORKING_DIR, 'input')
    if fname is None:
        return input_dir
    else:
        return os.path.join(WORKING_DIR, 'input', fname)

def write_df(df, fname='output_df', dir=None):
    if dir is not None:
        file_path = os.path.join(dir, fname)
    else:
        file_path=fname
    print(f'writing to the path: {file_path}')
    df.to_csv(file_path, index=False)


def read_df(fname='output_df', dir=None):
    if dir is not None:
        file_path = os.path.join(dir, fname)
    else:
        file_path=fname
    print(f'reading the dataframe: {file_path}')
    return pd.read_csv(file_path)

def select_dtype_df(df, dtype='NUMERIC'):
    if dtype == 'NUMERIC':
        return df.select_dtypes(include=NUMERICS_LIST)
    elif dtype == 'CATEGORICAL':
        # num_cols = df.select_dtypes(include=NUMERICS_LIST).columns
        cat_cols = df.columns[np.where(df.dtypes != np.float)[0]]
        # cat_cols = list(set(df.columns) - set(num_cols))
        return df[cat_cols]
    elif dtype == 'ALL':
        return df

cat_selector = make_column_transformer(
    ('passthrough', make_column_selector(dtype_include=['object','bool', np.int])), remainder='drop'
)

non_cat_selector = make_column_transformer(
    ('drop', make_column_selector(dtype_include=['object','bool', np.int])), remainder='passthrough'
)



non_num_selector = make_column_transformer(
    ('drop', make_column_selector(dtype_include=np.float)), remainder='passthrough'
)

num_selector = make_column_transformer(
    ('passthrough', make_column_selector(dtype_include=np.float)), remainder='drop'
)

def select_dtype_data(df, dtype:Union[str, list]='NUMERIC', return_type='pandas'):
    if isinstance(dtype, str):
        if dtype=='NUMERIC':
            num_ndarr = num_selector.fit_transform(df)
            if return_type=='pandas':
               return pd.DataFrame(num_ndarr, columns=num_selector.get_feature_names())
            else:
                return num_ndarr
        if dtype == 'CATEGORICAL':
            num_ndarr = cat_selector.fit_transform(df)
            if return_type == 'pandas':
                return pd.DataFrame(num_ndarr, columns=cat_selector.get_feature_names())
            else:
                return num_ndarr
    else:
        data_selector = make_column_transformer(
            ('passthrough', make_column_selector(dtype_include=dtype)), remainder='drop'
        )

        num_ndarr = data_selector.fit_transform(df)
        if return_type == 'pandas':
            return pd.DataFrame(num_ndarr, columns=data_selector.get_feature_names())
        else:
            return num_ndarr


def select_dtype_dataColumns(df, dtype:Union[str, list]='NUMERIC'):
    if isinstance(dtype, str):
        if dtype=='NUMERIC':
            num_selector.fit(df)
            return num_selector.get_feature_names()
        if dtype == 'CATEGORICAL':
            cat_selector.fit(df)
            return cat_selector.get_feature_names()
    else:
        data_selector = make_column_transformer(
            ('passthrough', make_column_selector(dtype_include=dtype)), remainder='drop'
        )
        data_selector.fit(df)
        return data_selector.get_feature_names()


def select_not_dtype_data(df, dtype:Union[str, list]='NUMERIC', return_type='pandas'):
    if isinstance(dtype, str):
        if dtype=='NUMERIC':
            num_ndarr = non_num_selector.fit_transform(df)
            if return_type=='pandas':
               return pd.DataFrame(num_ndarr, columns=non_num_selector.get_feature_names())
            else:
                return num_ndarr
        if dtype == 'CATEGORICAL':
            num_ndarr = non_cat_selector.fit_transform(df)
            if return_type == 'pandas':
                return pd.DataFrame(num_ndarr, columns=non_cat_selector.get_feature_names())
            else:
                return num_ndarr
    else:
        data_selector = make_column_transformer(
            ('drop', make_column_selector(dtype_include=dtype)), remainder='passthrough'
        )

        num_ndarr = data_selector.fit_transform(df)
        if return_type == 'pandas':
            return pd.DataFrame(num_ndarr, columns=data_selector.get_feature_names())
        else:
            return num_ndarr


def select_not_dtype_dataColumns(df, dtype:Union[str, list]='NUMERIC'):
    if isinstance(dtype, str):
        if dtype=='NUMERIC':
            num_selector.fit(df)
            return num_selector.get_feature_names()
        if dtype == 'CATEGORICAL':
            cat_selector.fit(df)
            return cat_selector.get_feature_names()
    else:
        data_selector = make_column_transformer(
            ('passthrough', make_column_selector(dtype_include=dtype)), remainder='drop'
        )
        data_selector.fit(df)
        return data_selector.get_feature_names()


def select_dtype_columns(df, dtype='NUMERIC', sub_dtype='ALL'):
    """
    :param df:
    :param dtype:
    :param sub_dtype:
    :return:
    """

    if dtype == 'NUMERIC':
        return df.select_dtypes(include=NUMERICS_LIST).columns
    elif dtype == 'CATEGORICAL':
        # num_cols = df.select_dtypes(include=NUMERICS_LIST).columns
        if sub_dtype == 'ALL':
            cat_cols = df.columns[np.where(df.dtypes != np.float)[0]]
        elif sub_dtype == 'object':
            cat_cols = df.columns[np.where(np.logical_and((df.dtypes != np.float).values, (df.dtypes != np.int).values))[0]]
        elif sub_dtype == 'int':
            cat_cols = df.columns[np.where(df.dtypes == np.int)[0]]

        return df[cat_cols].columns
    elif dtype == 'ALL':
        return df.columns


# def rename_columns(df, old_column_names, new_column_names):
#     if len(old_column_names) != len(new_column_names):
#         raise Exception(f'Should have the same length for old and new column names')
#
#     return df.rename(columns:)

def isNumeric(df):
    return df.iloc[:,0].dtype in NUMERICS_LIST
        # df.columns.values == df.select_dtypes(include=NUMERICS_LIST).columns.values


def get_columns_with_missing_values(df):
    nas = df.isna().sum()
    nas_df = pd.DataFrame(nas, columns=['null_count'])
    return np.array(nas_df[nas_df.null_count>0].index)


def missing_values_table(df):
        # Total missing values
        # print(self.df.head())


        mis_val = df.isnull().sum()
        n_rows = df.shape[0]
        # print(f'row_count - {n_rows}')

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / n_rows

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def normalize_dtype(source_df, target_df):
    """
    Change the target_df dtypes to same as source_df dataframes

    :param source_df: Pandas DataFrame. The dataframe to be used as reference.
    :param target_df: Pandas DataFrame. The dataframe that will be changed to same as source df
    :return: Pandas Dataframe. Which has the same datatypes as source_df
    """
    target_df = target_df.copy(deep=True)
    target_df = target_df.fillna(-99999)
    for col in target_df.columns:
        if col in source_df.columns:
            target_df[col] = target_df[col].astype(source_df[col].dtype)

    target_df = target_df.replace(to_replace=-99999, value=np.nan)

    return target_df


def convert_dtype(df, cols, dtype='float64'):
    """
    Summary: Converts the cols to certain dtype

    :param df: Pandas DataFrame
    :param cols: list or array-like: Columns to convert the dtype
    :param dtype: String
    :return: Pandas DataFrame
    """
    df = df.copy(deep=True)
    for col in cols:

        df[col] = df[col].astype(dtype)
    return df

def clean_col_name(col):
#     print(col)
    col = col.strip()
    if "'" in col:
        col = col.split("'")[1]
    return col