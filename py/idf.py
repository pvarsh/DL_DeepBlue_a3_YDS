import re
import csv
import math

file_in = "../data/train.csv"
file_out= "../idf/idf.csv"

print("Reading data...")
with open(file_in, 'r') as f:
    reader = csv.reader(f)
    data = [line for line in reader]

print("Initializing dicts...")
idf_dict = dict()

num_docs = float(650000)

regex = re.compile(r'[^a-zA-Z]')

print("Calculating word counts...")
for i, review in enumerate(data):
    review.append(set(re.sub(regex, " ", review[1].lower()).split()))

    for word in review[2]:
        if word in idf_dict:
            idf_dict[word] += 1
        else:
            idf_dict[word] = 1

print("Calculating idf...")
for word, count in idf_dict.iteritems():
    idf_dict[word] = math.log(num_docs/count)

print("Writing to file...")
with open(file_out, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(idf_dict.iteritems())
