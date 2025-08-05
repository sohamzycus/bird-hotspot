import csv
import json

with open('India Tehsil Centroid LatLong 1.csv', 'r', encoding='utf-8') as fin:
    reader = csv.DictReader(fin, delimiter='\t')
    data = list(reader)

with open('bird_hotspot.json', 'w', encoding='utf-8') as fout:
    json.dump(data, fout, indent=2)
