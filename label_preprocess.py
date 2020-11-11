import csv
import os

with open('dataset/training_labels.csv', 'r') as csvFile:
    cin = csv.reader(csvFile)
    training_labels = [label for label in cin]
    training_labels.pop(0)
    meta = set(label for (id, label) in training_labels)
    meta = sorted(meta)

if not os.path.isfile('dataset/label_seri_map_11.csv'):
    with open('dataset/label_seri_map_11.csv', 'w') as csvFile:
        cout = csv.writer(csvFile)
        cout.writerows([(seri, label) for seri, label in enumerate(meta)])

if not os.path.isfile('dataset/training_labels_seri_11.csv'):
    header = [('id', 'label', 'seri')]
    labels_seri_map = {label: seri for seri, label in enumerate(meta)}
    with open('dataset/training_labels_seri_11.csv', 'w') as csvFile:
        cout = csv.writer(csvFile)
        training_labels_seri = [(id, label, labels_seri_map[label])
                                for (id, label) in training_labels]
        cout.writerows(header + training_labels_seri)
