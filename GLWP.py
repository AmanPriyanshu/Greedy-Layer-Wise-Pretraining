import torch
import numpy as np
from tqdm import tqdm

class GLWP:
	def __init__(self, arr_neurons=[400, 256, 100], input_dims=784, classes=10, batch_size=64):
		self.arr_neurons = [input_dims]+arr_neurons
		self.input_dims = input_dims
		self.classes = classes
		self.batch_size = batch_size

	def get_single_layer_model(self, input_dims, hidden_dims):
		model = torch.nn.Sequential(
				torch.nn.Linear(input_dims, hidden_dims), 
				torch.nn.ReLU(), 
				torch.nn.Linear(hidden_dims, input_dims), 
				torch.nn.ReLU()
				)
		return model

	def train_single_layer(self, layer_num, dataset, epochs=2):
		model = self.get_single_layer_model(self.arr_neurons[layer_num], self.arr_neurons[layer_num+1])
		model.train()
		x = np.array([dataset[i*self.batch_size:(i+1)*self.batch_size] for i in range(len(dataset)//self.batch_size)], dtype=np.float32)
		x = torch.from_numpy(x)
		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
		for epoch in range(epochs):
			bar = tqdm(x)
			running_loss = []
			for batch_idx, batch in enumerate(bar):
				preds = model(batch)
				optimizer.zero_grad()
				loss = criterion(preds, batch)
				loss.backward()
				optimizer.step()
				running_loss.append(loss.item())
				bar.set_description(str({'epoch': epoch+1, 'running_loss': round(sum(running_loss)/len(running_loss), 4)}))
			bar.close()

		with torch.no_grad():
			weight = model[0].weight.detach().numpy()
			bias = model[0].bias.detach().numpy()
			print(type(weight), type(bias))

if __name__ == '__main__':
	import pandas as pd
	df_train = pd.read_csv("./data/MNIST.csv")
	df_train = df_train.values
	y = df_train.T[0]
	df_train = df_train.T[1:].T
	df_test = df_train[40000:]
	df_train = df_train[:40000]
	glwp = GLWP()
	glwp.train_single_layer(0, df_train/255)