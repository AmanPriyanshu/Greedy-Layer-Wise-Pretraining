import torch
import numpy as np
from tqdm import tqdm
import cv2

class GLWP:
	def __init__(self, arr_neurons=[400, 256, 100], input_dims=784, classes=10, batch_size=64, epochs=10):
		self.arr_neurons = [input_dims]+arr_neurons
		self.input_dims = input_dims
		self.classes = classes
		self.batch_size = batch_size
		self.epochs = epochs
		self.models = []
		self.final_model = None
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def get_untrained_model(self):
		models = []
		for idx in range(len(self.arr_neurons)-1):
			models.append(self.get_single_layer_model_forward(self.arr_neurons[idx], self.arr_neurons[idx+1]))
		model = torch.nn.Sequential(torch.nn.Linear(self.arr_neurons[-1], self.classes))
		models.append(model)
		model_list = list(models[0].children())
		for model in models[1:]:
			model_list.extend(list(model.children()))
		final_model = torch.nn.Sequential(*model_list)
		return final_model

	def get_single_layer_model_forward(self, input_dims, hidden_dims):
		model = torch.nn.Sequential(
				torch.nn.Linear(input_dims, hidden_dims), 
				torch.nn.ReLU(),
				)
		return model

	def get_single_layer_model(self, input_dims, hidden_dims):
		model = torch.nn.Sequential(
				torch.nn.Linear(input_dims, hidden_dims), 
				torch.nn.ReLU(), 
				torch.nn.Linear(hidden_dims, self.classes)
				)
		return model

	def train_single_layer(self, layer_num, labels, dataset):
		model = self.get_single_layer_model(self.arr_neurons[layer_num], self.arr_neurons[layer_num+1])
		model = model.to(self.device)
		model.train()
		x = np.array([dataset[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(dataset)//self.batch_size)], dtype=np.float32)
		x = torch.from_numpy(x)
		y = np.array([labels[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(dataset)//self.batch_size)])
		y = torch.from_numpy(y)
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
		for epoch in range(self.epochs):
			bar = tqdm(zip(x, y), total=len(x))
			running_loss, running_acc = [], []
			for batch_idx, (batch_x, batch_y) in enumerate(bar):
				batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
				outputs = model(batch_x)
				optimizer.zero_grad()
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()
				running_loss.append(loss.cpu().item())
				preds = torch.argmax(outputs, 1)
				acc = torch.mean((preds==batch_y).float())
				running_acc.append(acc.cpu().item())
				bar.set_description(str({'epoch': epoch+1, 'running_loss': round(sum(running_loss)/len(running_loss), 4), 'running_acc': round(sum(running_acc)/len(running_acc), 4)}))
			bar.close()

		with torch.no_grad():
			weight = model[0].weight.detach().cpu().numpy()
			bias = model[0].bias.detach().cpu().numpy()
			return weight, bias

	def generate_dataset(self, layer_index, dataset):
		if layer_index==0:
			return dataset
		x = np.array([dataset[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(dataset)//self.batch_size)], dtype=np.float32)
		x = [torch.from_numpy(i) for i in x]
		for idx in range(layer_index):
			model = self.models[idx]
			model.eval()
			model = model.to(self.device)
			for batch_idx, batch in enumerate(tqdm(x, desc="Generating - Layer: "+str(idx+1))):
				batch = batch.to(self.device)
				preds = model(batch)
				x[batch_idx] = preds.detach()
		x = [i.cpu().numpy() for i in x]
		return np.concatenate(x, axis=0)

	def train(self, labels, dataset):
		for layer_index in range(len(self.arr_neurons)-1):
			w, b = self.train_single_layer(layer_index, labels, self.generate_dataset(layer_index, dataset))
			model = self.get_single_layer_model_forward(self.arr_neurons[layer_index], self.arr_neurons[layer_index+1])
			model[0].weight = torch.nn.Parameter(torch.from_numpy(w))
			model[0].bias = torch.nn.Parameter(torch.from_numpy(b))
			self.models.append(model)
		model = torch.nn.Sequential(torch.nn.Linear(self.arr_neurons[-1], self.classes))
		self.models.append(model)
		model_list = list(self.models[0].children())
		for model in self.models[1:]:
			model_list.extend(list(model.children()))
		self.final_model = torch.nn.Sequential(*model_list)

	def train_classifier(self, labels, dataset, model):
		x = np.array([dataset[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(dataset)//self.batch_size)], dtype=np.float32)
		x = torch.from_numpy(x)
		y = np.array([labels[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(dataset)//self.batch_size)])
		y = torch.from_numpy(y)
		criterion = torch.nn.CrossEntropyLoss()
		print("Training Classifiers...")
		optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
		model.train()
		model = model.to(self.device)
		for epoch in range(self.epochs):
			bar = tqdm(zip(x, y), total=len(x))
			running_loss, running_acc = [], []
			for batch_idx, (batch_x, batch_y) in enumerate(bar):
				batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
				outputs = model(batch_x)
				optimizer.zero_grad()
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()
				running_loss.append(loss.cpu().item())
				preds = torch.argmax(outputs, 1)
				acc = torch.mean((preds==batch_y).float())
				running_acc.append(acc.cpu().item())
				bar.set_description(str({'epoch': epoch+1, 'running_loss': round(sum(running_loss)/len(running_loss), 4), 'running_acc': round(sum(running_acc)/len(running_acc), 4)}))
			bar.close()


def main_mnist():
	import pandas as pd
	df_train = pd.read_csv("./data/MNIST.csv")
	df_train = df_train.values
	y = df_train.T[0]
	df_train = df_train.T[1:].T
	df_test = df_train[40000:]
	y_test = y[40000:]
	df_train = df_train[:40000]
	y_train = y[:40000]
	glwp = GLWP()
	scaled = df_train.copy()/255

	print("\nNot Pre-Trained")
	model = glwp.get_untrained_model()
	glwp.train_classifier(y_train, scaled, model)

	print("\nPre-Trained")
	glwp.train(y_train, scaled)
	model = glwp.final_model
	glwp.train_classifier(y_train, scaled, model)

def main_cifar():
	from data.CIFAR.load_cifar import load_data
	train_x, train_y, test_x, test_y = load_data("./data/CIFAR/")
	train_x = np.reshape(train_x, (len(train_x), -1))
	glwp = GLWP(arr_neurons=[2048, 516, 100], input_dims=3072, epochs=20)
	scaled = train_x.copy()/255

	print("\nNot Pre-Trained")
	model = glwp.get_untrained_model()
	glwp.train_classifier(train_y.astype(np.int64), scaled, model)

	print("\nPre-Trained")
	glwp.train(train_y.astype(np.int64), scaled)
	model = glwp.final_model
	glwp.train_classifier(train_y.astype(np.int64), scaled, model)

if __name__ == '__main__':
	main_mnist()