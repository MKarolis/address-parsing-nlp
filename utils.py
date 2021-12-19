from numpy import string_
import pandas as pd

def read_DataFrame_from_excel(filename: str, numberOfRows: int = None):
    """
    Function that reads data from a excel file and put it in a dataset.

    Args:
        filename (str): name of the file to be read
        numberOfRows (int): numbers of rows to be read

    Returns:
        dataset: dataset with all the data read from the excel file
    """
    return pd.read_excel(filename, nrows = numberOfRows, keep_default_na=False)


def read_dataFrame_from_csv(filename: str):
    """
    Function that reads data from a txt file and put it in a dataset.

    Args:
        filename (str): name of the file to be read

    Returns:
        dataset: dataset with all the data read from the txt file
    """
    return pd.read_csv(filename, delimiter='\t', keep_default_na=False)


def write_DataFrame_to_excel(df: pd.DataFrame, filename: str):
    """
    Function that writes data from a dataset to a excel file.

    Args:
        df (pd.DataFrame): dataset to be written to the file
        filename (str): name to be given to the new file   
    """
    sheet_name = 'Output'

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        # format all data as a table
        worksheet.add_table(0, 0, df.shape[0], df.shape[1]-1, {
            'columns': [{'header': col_name} for col_name in df.columns],
            'style': 'Table Style Medium 5'
        })


def resolve_model_name(region: str = 'DEFAULT'):
    """
    Resolves trained model's name by appending the region suffix

    Args:
        region - region suffix, defaults to 'DEFAULT'
    """
    return 'trained_model_{}'.format(region)
