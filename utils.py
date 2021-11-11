import pandas as pd

def read_DataFrame_from_file(filename: str, numberOfRows: int = None):
    return pd.read_excel(filename, nrows = numberOfRows, keep_default_na=False)


def write_DataFrame_to_excel(df: pd.DataFrame, filename: str):
    sheet_name = 'Output'

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        # format all data as a table
        worksheet.add_table(0, 0, df.shape[0], df.shape[1]-1, {
            'columns': [{'header': col_name} for col_name in df.columns],
            'style': 'Table Style Medium 5'
        })
