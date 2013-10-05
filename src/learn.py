import csv

def file_reader(file):
    reader = csv.reader(utf_8_encoder(file), quotechar = '"')
    for row in reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def utf_8_encoder(csv):
    with open(csv, 'r') as text:
        for line in text:
            yield line.encode('utf-8')


file = file_reader('../data/train-utf8.csv')
print file.next()
print file.next()
print file.next()
print file.next()
