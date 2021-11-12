from tokenize import tokenize

import pandas as pd

before_data = [
    '2-15, Meiwadori 3-chome, Hyogo-ku, Kobe-shi, Hyogo 652-0882',
    '12-4, Sagisu 5-chome, Fukushima-ku, Osaka-shi, Osaka; 5530 002',
]

train_data = [
    ('2-15, Meiwadori 3-chome, Hyogo-ku, Kobe-shi, Hyogo 652-0882', {
        'entities': [(0, 4, 'street'), (6, 23, 'street'), (25, 33, 'area'), (51, 59, 'postal'), (35, 43, 'city'),
                     (25, 30, 'region')]}),
    ('12-4, Sagisu 5-chome, Fukushima-ku, Osaka-shi, Osaka; 5530 002', {
        'entities': [(0, 4, 'street'), (6, 20, 'street'), (22, 34, 'area'), (54, 62, 'postal'), (36, 45, 'city'),
                     (36, 41, 'region')]}),
]

import pandas as pd
import re


def read_DataFrame_from_file(filename: str, numberOfRows: int = None):
    return pd.read_excel(filename, nrows=numberOfRows, keep_default_na=False)


# %%

DATA_INPUT_FILENAME = 'training_data.xlsx'
NUMBER_OF_PARSABLE_RECORDS = 999


def preprocess_data(data: pd.DataFrame):
    data['person_address'] = data.apply(lambda row: re.sub(r'([^\s])(,)([^\s])', r'\1, \3', row['person_address']),
                                        axis=1)
    print(data['person_address'][44][1])


raw_data: pd.DataFrame = read_DataFrame_from_file(DATA_INPUT_FILENAME, NUMBER_OF_PARSABLE_RECORDS)
preprocess_data(raw_data)

# %%

import re

token_types: set = {'co', 'building', 'street', 'nr', 'area', 'postal', 'city', 'region', 'country'}


def get_entity_list(entry: dict, address: str):
    # print(entry)
    entities: list = []
    present_tokens = filter(lambda item: item[0] in token_types and item[1] and str(item[1]).strip(), entry.items())
    test = address
    for item in present_tokens:
        token_value = str(item[1]).strip()
        match = re.search(re.escape(token_value), test)
        if match:
            span = match.span()
            fromEntry = address[span[0]:span[1]]
            times = address.count(fromEntry)
            if times > 1:
                tkn = address.split()
                print('tkn: ', type(tkn))
                print(fromEntry, ' - count times: ', times, ' item:', item, ' add: ', address)

                size = -1
                for i in tkn:
                    if str(i).endswith(str(item[1])) and str(i).startswith(str(item[1])):
                        firstIdx = size + 1
                        lastIdx = firstIdx + len(str(i)) - 1
                        print('item: ', item[1], ' first: ', firstIdx, ' last: ', lastIdx)
                        entities.append((firstIdx, lastIdx, item[0]))

                    string = str(i)
                    length = len(string) - 1
                    size = size + length + 2
                    print(address, 'size: ', size, ' len: ', len(string))


            else:
                entities.append((span[0], span[1], item[0]))
        else:
            # Try and resolve multiple tokens separated by ';'
            split_items = map(lambda token: token.strip(), token_value.split(';'))
            for token in split_items:
                split_match = re.search(re.escape(token), test)
                if split_match:
                    span = split_match.span()
                    fromEntry = address[span[0]:span[1]]
                    times = address.count(fromEntry)
                    if times > 1:
                        tkn = address.split()
                        print('tkn: ', type(tkn))
                        print(fromEntry, ' - count times: ', times, ' item:', item, ' add: ', address, ' alone:')

                        size = -1
                        for i in tkn:
                            if str(i).endswith(str(item[1])) and str(i).startswith(str(item[1])):
                                firstIdx = size + 1
                                lastIdx = firstIdx + len(str(i)) - 1
                                print('item: ', item[1], ' first: ', firstIdx, ' last: ', lastIdx)
                                entities.append((firstIdx, lastIdx, item[0]))

                            string = str(i)
                            length = len(string) - 1
                            size = size + length + 2
                            print(address, 'size: ', size, ' len: ', len(string))

                    else:
                        entities.append((span[0], span[1], item[0]))
                    # entities.append((span[0], span[1], item[0]))
                else:
                    # address.replace(token,'',1)
                    print('WARNING: could not find token "{}" in address "{}"'.format(token, address))

    return entities


# def removeOverlaps(token, entry):


def map_to_training_entry(entry: dict):
    address = entry['person_address']
    # print(entry)
    return (address, {
        'entities': get_entity_list(entry, address)
    })


train_data = list(
    map(map_to_training_entry, raw_data.to_dict('records'))
)


def entities_overlap(entry):
    entities = entry[1]['entities']
    # print(entry[1]['entities'])
    for first in entities:
        # print('first:', first)
        for second in entities:
            # print('second:', second)
            if (first == second): continue
            if (first[0] < second[0] < first[1]) or (first[0] > second[0] > first[1]) or (
                    first[0] == second[0] or first[1] == second[1]):
                print('Entities {} and {} overlap in "{}"'.format(first, second, entry[0]))
                return True
    return False


train_data = list(filter(lambda entry: not entities_overlap(entry), train_data))
