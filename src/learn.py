import csv
import codecs
import pandas as pd

def load_data_frame(path):

    def unicode_csv_reader(unicode_csv_data):
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data))
        for row in csv_reader:
            yield [unicode(cell, 'utf-8') for cell in row]

    def utf_8_encoder(unicode_csv_data):
        for line in unicode_csv_data:
            yield line.encode('utf-8')

    file = unicode_csv_reader(codecs.open(path, 'r', 'utf-8'))

    columns = next(file)
    return pd.DataFrame.from_records([x for x in file], columns=columns)

insults = load_data_frame('../data/train-utf8.csv')
