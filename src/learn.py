import csv
import codecs
import pandas as pd

def unicode_csv_reader(unicode_csv_data):
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data))
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

file = unicode_csv_reader(codecs.open('../data/train-utf8.csv', 'r', 'utf-8'))
columns = next(file)
df = pd.DataFrame.from_records([x for x in file], columns=columns)
