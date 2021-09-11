# Greedy-Layer-Wise-Pretraining
Training DNNs are normally memory and computationally expensive. Therefore, we explore greedy layer-wise pretraining.

# Images:

## Supervised:

![supervised](/images/supervised.png)

## Unsupervised:

![unsupervised](/images/unsupervised.png)

## Without vs With Unsupervised Pre-Training : CIFAR

 
| Without | Pre- | Training | | With | Pre- | Training |
| ----- | ------ | -------- |-| ----- | ------ | -------- |
| Epoch | Loss   | Accuracy | | Epoch | Loss   | Accuracy |
| ----- | ------ | -------- |-| ----- | ------ | -------- |
| 1     | 2.204  | 0.1872   | | 1     | 1.9574 | 0.2998   |
| 2     | 1.9746 | 0.2861   | | 2     | 1.7756 | 0.3731   |
| 3     | 1.8704 | 0.3281   | | 3     | 1.7076 | 0.3986   |
| 4     | 1.803  | 0.3534   | | 4     | 1.654  | 0.4184   |
| 5     | 1.743  | 0.3769   | | 5     | 1.6064 | 0.4335   |
| 6     | 1.6938 | 0.3929   | | 6     | 1.5636 | 0.4481   |
| 7     | 1.6509 | 0.409    | | 7     | 1.5256 | 0.4608   |
| 8     | 1.6117 | 0.4234   | | 8     | 1.4915 | 0.4715   |
| 9     | 1.5762 | 0.4355   | | 9     | 1.461  | 0.4813   |
| 10    | 1.544  | 0.447    | | 10    | 1.433  | 0.4916   |


## Without vs With Supervised Pre-Training : CIFAR

 
| Without | Pre- | Training | | With | Pre- | Training |
| ----- | ------ | -------- |-| ----- | ------ | -------- |
| Epoch | Loss   | Accuracy | | Epoch | Loss   | Accuracy |
| ----- | ------ | -------- |-| ----- | ------ | -------- |
| 1     | 2.204  | 0.1872   | | 1     | 1.1234 | 0.6031   |
| 2     | 1.9746 | 0.2861   | | 2     | 1.0139 | 0.6436   |
| 3     | 1.8704 | 0.3281   | | 3     | 0.969 | 0.66   |
| 4     | 1.803  | 0.3534   | | 4     | 0.932  | 0.6735   |
| 5     | 1.743  | 0.3769   | | 5     | 0.8983 | 0.6855   |
| 6     | 1.6938 | 0.3929   | | 6     | 0.865 | 0.6993   |
| 7     | 1.6509 | 0.409    | | 7     | 0.8364 | 0.7105   |
| 8     | 1.6117 | 0.4234   | | 8     | 0.8034 | 0.7219   |
| 9     | 1.5762 | 0.4355   | | 9     | 0.7789 | 0.73   |
| 10    | 1.544  | 0.447    | | 10    | 0.7514  | 0.7409   |











