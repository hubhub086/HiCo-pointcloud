from sklearn.cluster import kmeans_plusplus
import numpy as np

data = np.load('./cookbook_trainData_64.npy', encoding='bytes', allow_pickle=True)

print('==> fitting <==')
centers_init, indices = kmeans_plusplus(data, n_clusters=8192, random_state=0)
print('==> finish k-means++ <==')
np.save('./cookbook-train.npy', centers_init)
print(centers_init.shape)
print(indices)
