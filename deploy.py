import pandas as pd
import spacy
from train import TRAINED_MODEL_FILENAME, preprocess_data
from utils import read_dataFrame_from_csv, read_DataFrame_from_excel, write_DataFrame_to_excel


PARSED_DATA_FILENAME = './files/parsed.xlsx'


def enrich_row_with_address_details(row, nlp):
    """
    Gets address properties from a DataFrame row using a NLP model
    
    Args:
        row: An object having a 'person_address' property
            Example:
            row = {
                'person_address': 'Ozo g. 25, Vilnius',
                'city': 'Vilnius',
                'street': 'Ozo g.',
                'nr': 25
            }
        nlp: A spacy NLP model

    Returns:
        an array indicating address's co, building, street, nr, area, postal, city, region, country in this order
    """

    obj = {
        'co': '',
        'building': '',
        'street': '',
        'nr': '',
        'area': '',
        'postal': '',
        'city': '',
        'region': '',
        'country': ''
    }

    doc = nlp(row['person_address'])
    for ent in doc.ents:
        if (len(obj[ent.label_])):
            obj[ent.label_] = '{}; {}'.format(obj[ent.label_], ent.text)
        else:
            obj[ent.label_] = ent.text

    return [
        obj['co'],
        obj['building'],
        obj['street'],
        obj['nr'],
        obj['area'],
        obj['postal'],
        obj['city'],
        obj['region'],
        obj['country']
    ]


def parse_addresses(frame: pd.DataFrame):
    """
    Parses addresses in a given DataFrame by adding co, building, street, nr, area, postal, city, region, country information
    
    Args:
        frame (DataFrame): A data frame having a 'person_address' property. This argument is not mutated by the code

    Returns:
        A DataFrame having extracted properties
    """

    data: pd.DataFrame = frame.copy()
    original_addresses = data['person_address']
    preprocess_data(data)

    MODEL_BY_COUNTRY = {
        'JP': spacy.load('./models/trained_model_JP')
    }
    default_nlp: spacy.language = spacy.load('./models/trained_model_DEFAULT')

    data[['co', 'building', 'street', 'nr', 'area', 'postal', 'city', 'region', 'country']] = data.apply(
        lambda row: enrich_row_with_address_details(
            row, 
            MODEL_BY_COUNTRY[row['person_ctry_code']] if row['person_ctry_code'] in MODEL_BY_COUNTRY else default_nlp
        ),
        axis=1,
        result_type='expand'
    )
    data['person_address'] = original_addresses
    return data


if __name__ == '__main__':

    filename: str = input('Enter the path of the file with the address data that you wish to parse (e.g. ./files/input.xlsx):\n> ')
    
    data: pd.DataFrame = read_DataFrame_from_excel(filename) if filename.endswith('.xlsx') else read_dataFrame_from_csv(filename)
    data = parse_addresses(data)

    write_DataFrame_to_excel(data, PARSED_DATA_FILENAME)
