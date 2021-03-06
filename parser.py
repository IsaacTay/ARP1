import csv
import numpy as np

# returns [(label, 28*28 mono image)]
def get_data(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        first = True
        data = []
        for row in reader:
            if first:
                first = False
                continue
            data.append((row[0], np.array(row[1:]).reshape(28, 28)))
        return data
