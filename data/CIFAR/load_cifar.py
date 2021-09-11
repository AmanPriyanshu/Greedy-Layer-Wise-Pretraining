import pickle
import numpy as np
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(path):
  X, Y = None, None
  for i in [path+i for i in sorted(os.listdir(path)) if '.' not in i and '__pycache__' not in i]:
    print(i)
    d = unpickle(i)
    y = d[b'labels']
    x = d[b'data']
    if X is None:
      X, Y = x, y
    else:
      Y = Y + y
      X = np.concatenate((X, x), 0)
  train_x = X[:50000]
  train_y = Y[:50000]
  test_x = X[50000:]
  test_y = Y[50000:]
  train_x = np.reshape(train_x, (-1, 3, 32, 32))
  test_x = np.reshape(test_x, (-1, 3, 32, 32))
  train_x = np.stack([np.transpose(np.reshape(image, (3, 32,32)), (1,2,0)) for image in train_x])
  test_x = np.stack([np.transpose(np.reshape(image, (3, 32,32)), (1,2,0)) for image in test_x])
  return train_x, np.array(train_y), test_x, np.array(test_y)

'''
train_x, train_y, test_x, test_y = load_data("./data/CIFAR/")
  for i in range(10):
    indexes = np.argwhere(train_y==i).flatten()
    data = train_x[indexes]
    for idx, image in tqdm(enumerate(data), total=len(data)):
      output_path = "./cifar/"+str(i)+"/"+"0"*(len(str(len(data))) - len(str(idx)))+str(idx)+".png"
      cv2.imwrite(output_path, image)
'''