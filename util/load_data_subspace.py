from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

class DataSet(Dataset):
	
	def __init__(self, fold_number: int, train: bool = True):
		
		f = loadmat('./data/ETH-80_d10.mat')
		
		g = loadmat('./data/ETH-80_traintest.mat')
		self.train_test_split = g['ids']
		
		self._train = train
		self._fold_number = fold_number 
		
		if self._train:
			self.images = f['dataset'][self.train_test_split[self._fold_number, 0]][0] # or 'data'
			self.labels = f['labels'][self.train_test_split[self._fold_number, 0]][0]
		else:
			self.images = f['dataset'][self.train_test_split[self._fold_number, 1]][0] # or 'data'
			self.labels = f['labels'][self.train_test_split[self._fold_number, 1]][0]


	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):		
			
		return torch.from_numpy(self.images[index]), torch.from_numpy(self.labels[index]).T[0]


# if __name__=='__main__':
# 	d = DataSet()
