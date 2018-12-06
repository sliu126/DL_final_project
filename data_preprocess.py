import csv
from random import shuffle
import pickle

def read_csv_data(csv_file_name):
	data_all = {}
	data_all["train"] = []
	data_all["dev"] = []
	data_all["test"] = []
	csv_reader = csv.reader(open(csv_file_name), delimiter=',')
	line_count = 0
	cat_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3} # make categories represented by number
	for row in csv_reader:
		if line_count == 0:
			# skip the first line
			line_count += 1
		else:
			title = row[1]
			category = row[4]
			data_entry = {"title": title, "cat": cat_dict[category]}
			# 80% training data, 10% dev data, 10% test data (can change this)
			if line_count % 10 < 8:
				data_all["train"].append(data_entry)
			elif line_count % 10 == 8:
				data_all["dev"].append(data_entry)
			else:
				data_all["test"].append(data_entry)
			line_count += 1
	shuffle(data_all["train"])
	shuffle(data_all["dev"])
	shuffle(data_all["test"])
	return data_all


if __name__ == '__main__':
	data_all = read_csv_data("uci-news-aggregator.csv")
	pickle.dump(data_all, open("data_all.pkl", "wb"))


