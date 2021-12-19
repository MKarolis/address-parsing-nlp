import pandas as pd
import re
import spacy
import random
from utils import read_DataFrame_from_excel
from spacy.util import compounding, minibatch


TRAINING_DATA_FILENAME = 'training_data_fixed.xlsx'
TRAINING_ENTRIES_COUNT = 999
TRAINED_MODEL_FILENAME = 'trained_model'

TOKEN_TYPES: set = {'co', 'building', 'street', 'nr', 'area', 'postal', 'city', 'region', 'country'}

TRAIN_ITERATION_COUNT = 25
TRAIN_DROP_PROPERTY = 0.5


def preprocess_data(data: pd.DataFrame):
    """
    Performs data preprocessing by adding a space after each comma and semicolon if they are missing
    
    Args:
        dataFrame (pd.DataFrame): dataset to be processed

    Returns:
        None
    """
    for col in data.columns:
        data[col] = data.apply(lambda row: re.sub(r'([^\s])([,;])([^\s])', r'\1\2 \3', str(row[col])), axis=1)


def get_entity_list(entry: dict, adr: str):
    """
    Extracts an array of tuples, indicating positions of tokens in a provided address
    
    Args:
        entry (dict): dictionary, where keys are token types.
            Example:
            dict = {
                'city': 'Vilnius',
                'street': 'Ozo g.',
                'nr': 25
            }
        adr (str): an address string.
            Example: 
            adr = 'Ozo g. 25, Vilnius'

    Returns:
        Array of tuples, where tuples follow structure of (token_position_start, token_position_end, token)
    """
    address = str(adr)
    entities: list = []
    present_tokens = filter(lambda item: item[0] in TOKEN_TYPES and item[1] and str(item[1]).strip(), entry.items())

    ## tokens to retry matching
    retry_tokens: set = set()

    for item in present_tokens:
        token_value = str(item[1]).strip()
        match = re.search(re.escape(token_value), address)
        if match:
            # If multiple occurences can be matched, save the token to be matched later
            if (len(re.findall(re.escape(token_value), address)) > 1):
                retry_tokens.add((token_value, item[0]))
                continue
            span = match.span()
            entities.append((span[0], span[1], item[0]))
            # Replace matched entity with symbols, so that parts of it cannot be matched again
            address = address[:span[0]] + '$' * (span[1] - span[0]) + address[span[1]:]
        else:
            # Try and resolve multiple tokens separated by ';'
            split_items = map(lambda token: token.strip(), token_value.split(';'))
            for token in split_items:
                split_match = re.search(re.escape(token), address)
                if split_match:
                    # If multiple occurences can be matched, save the token to be matched later
                    if (len(re.findall(re.escape(token), address)) > 1):
                        retry_tokens.add((token, item[0]))
                        continue
                    span = split_match.span()
                    entities.append((span[0], span[1], item[0]))
                    # Replace matched entity with symbols, so that parts of it cannot be matched again
                    address = address[:span[0]] + '$' * (span[1] - span[0]) + address[span[1]:]
                else:
                    print('WARNING: could not find token "{}" in address "{}"'.format(token, adr))
    
    # Try and match previously marked tokens, now that single-match entities were eliminated
    for token, tkn_type in retry_tokens:
        token_value = str(token).strip()
        match = re.search(re.escape(token_value), address)
        if match:
            span = match.span()
            entities.append((span[0], span[1], tkn_type))
            address = address[:span[0]] + '$' * (span[1] - span[0]) + address[span[1]:]
        else:
            print('WARNING: could not find token "{}" in address "{}"'.format(token, adr))

    return entities


def map_to_training_entry(entry: dict):
    """
    Maps an object of address tokens into a tuple of address string and an object containing entity list.
    
    Args:
        entry (dict): dictionary, where keys include token types.
            Example:
            dict = {
                'person_address': 'Ozo g. 25, Vilnius',
                'city': 'Vilnius',
                'street': 'Ozo g.',
                'nr': 25
            }

    Returns:
        A tuple, where first element is the address, and the second one is an object containing the entity list
    """
    address = entry['person_address']
    return (address, {
        'entities': get_entity_list(entry, address)
    })


def entities_overlap(entry):
    """
    Checks whether an entry contains overlapping entities
    
    Args:
        entry (array or tuple): dictionary, where keys are token types.
            Example:
            dict = {
                'city': 'Vilnius',
                'street': 'Ozo g.',
                'nr': 25
            }
        adr (str): an address string.
            Example: 
            adr = 'Ozo g. 25, Vilnius'

    Returns:
        Array of tuples, where tuples follow structure of (token_position_start, token_position_end, token)
    """
    entities = entry[1]['entities']
    for first in entities:
        for second in entities:
            if (first == second): continue
            if (first[0] < second[0] and first[1] > second[0]) or (first[0] > second[0] and first[1] < second[0]) or (first[0]==second[0] or first[1]==second[1]):
                print('Entities {} and {} overlap in "{}"'.format(first, second, entry[0]))
                return True
    return False


if __name__ == '__main__':

    raw_data: pd.DataFrame = read_DataFrame_from_excel(TRAINING_DATA_FILENAME, TRAINING_ENTRIES_COUNT)
    preprocess_data(raw_data)

    train_data = map(map_to_training_entry, raw_data.to_dict('records'))
    train_data = list(filter(lambda entry: not entities_overlap(entry), train_data))

    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)

    for token in TOKEN_TYPES:
        ner.add_label(token)
    
    print('--- TRAINING THE MODEL IN {} ITERATIONS | DROP = {} ---'.format(TRAIN_ITERATION_COUNT, TRAIN_DROP_PROPERTY))
    optimizer = nlp.begin_training()

    for itn in range(TRAIN_ITERATION_COUNT):
        random.shuffle(train_data)
        losses = {}

        batches = minibatch(train_data, size=compounding(4, 32, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  
                annotations,  
                drop=TRAIN_DROP_PROPERTY,  
                sgd=optimizer,
                losses=losses)
        print('Iteration: {} | Losses: {}'.format(itn, losses))

    nlp.to_disk(TRAINED_MODEL_FILENAME)
