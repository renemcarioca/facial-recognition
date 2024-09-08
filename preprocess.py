import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
from typing import Optional


class Preprocess:
	pairs_dev_train_path = 'pairsDevTrain.txt'
	pairs_dev_test_path = 'pairsDevTest.txt'
	people_dev_train_path = 'peopleDevTrain.txt'
	people_dev_test_path = 'peopleDevTest.txt'
	dataset_path = 'lfw'

	@staticmethod
	def get_image_path(name, num, img_dir = dataset_path):
		return f'{img_dir}/{name}/{name}_{num:0>4}.jpg'
	
	@staticmethod
	def get_image(name, num, img_dir = dataset_path):
		return Image.open(Preprocess.get_image_path(name, num, img_dir)).convert("RGB")

	@staticmethod
	def extract_pairs(filepath: str):
		pairs = []
		with open(filepath, 'r') as f:
			size = int(next(f))
			for i in range(size):
				line = next(f)
				name, num1, num2 = line.strip().split()
				pairs.append(((name, int(num1)), (name, int(num2))))
			for i in range(size):
				line = next(f)
				name1, num1, name2, num2 = line.strip().split()
				pairs.append(((name1, int(num1)), (name2, int(num2))))
		return pairs
	
	@staticmethod
	def extract_people(filepath: str):
		people = {}
		with open(filepath, 'r') as f:
			size = int(next(f))
			for i in range(size):
				line = next(f)
				name, num = line.strip().split()
				people[name] = int(num)
		return people
	
	@staticmethod
	def load_train_pairs(transform=None):
		train_pairs = Preprocess.extract_pairs(Preprocess.pairs_dev_train_path)
		return PairDataGenerator(train_pairs, transform=transform)

	@staticmethod
	def load_test_pairs(transform=None):
		test_pairs = Preprocess.extract_pairs(Preprocess.pairs_dev_test_path)
		return PairDataGenerator(test_pairs, transform=transform)
	
	@staticmethod
	def load_sample_pairs(size: int = 20, transform=None):
		train_pairs = Preprocess.extract_pairs(Preprocess.pairs_dev_train_path)
		rng = np.random.default_rng()
		rng.shuffle(train_pairs)
		return PairDataGenerator(train_pairs[:size], transform=transform)
	
	@staticmethod
	def load_train_people(shuffle=False, seed: Optional[int] = None, transform=None):
		train_people = Preprocess.extract_people(Preprocess.people_dev_train_path)
		return ImageGenerator(train_people, transform=transform, shuffle=shuffle, seed=seed)

	@staticmethod
	def load_test_people():
		test_people = Preprocess.extract_people(Preprocess.people_dev_test_path)
		return ImageGenerator(test_people)
		
		
class ImageGenerator(IterableDataset):
	def __init__(self, 
			people, 
			img_dir = Preprocess.dataset_path, 
			transform=None, 
			output_name=False, 
			shuffle=False,
			seed: Optional[int] =None
		):
		self.people = people
		self.people_order = list(people.keys())
		self.img_dir = img_dir
		self.transform = transforms.ToTensor() if transform is None else transform
		self.output_name = output_name
		self.shuffle = shuffle
		self.rng = np.random.default_rng(seed=seed)
	
	def __len__(self):
		return np.sum(list(self.people.values()))
	
	def __iter__(self):
		if self.shuffle:
			self.rng.shuffle(self.people_order)
		for name in self.people_order:
			image_count = self.people[name]
			image_order = np.arange(1, image_count + 1)
			if self.shuffle:
				self.rng.shuffle(image_order)
			for num in image_order:
				image = Preprocess.get_image(name, num, img_dir=self.img_dir)
				image = self.transform(image)
				if self.output_name:
					yield image, name
				else:
					yield image

class PairDataGenerator(Dataset):
	def __init__(self, pairs, img_dir = Preprocess.dataset_path, transform=None):
		self.pairs = pairs
		self.img_dir = img_dir
		self.transform = transforms.ToTensor() if transform is None else transform

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, index):
		(name1, num1), (name2, num2) = self.pairs[index]
		# Load images
		image1 = Preprocess.get_image(name1, num1, img_dir=self.img_dir)
		image2 = Preprocess.get_image(name2, num2, img_dir=self.img_dir)
		# Transform images
		image1 = self.transform(image1)
		image2 = self.transform(image2)
		# Create label
		y_true = 1 if name1 == name2 else 0
		return image1, image2, y_true