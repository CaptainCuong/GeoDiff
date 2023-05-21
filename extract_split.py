import pickle

dir_ = './data/GEOM/QM9/test_data_200.pkl'
with open(dir_, 'rb') as f:
	data_graph = pickle.load(f)
	
test_data = [data.smiles for data in data_graph]
print(test_data)