import pickle
import yaml

# Save object into compressed pickle file
def save_as_pkl(object, path):
	pickle.dump(object, open(path, "wb"))

# Load from pickle file
def load_pkl(path):
	return pickle.load(open(path, "rb"))

def read_yaml():
	filename = "input.yaml"
	in_file = open(filename, 'r')
	data = in_file.read()
	config = yaml.load(data)
	return config