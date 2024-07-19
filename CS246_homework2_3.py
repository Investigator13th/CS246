# 探究学习率的影响
# 探究初始化值的影响

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

# 加载数据
data = pd.read_csv('data/ratings.train.txt', sep='\t', names=['user_id', 'item_id', 'rating'])

# 初始化矩阵
num_users = data['user_id'].max() + 1
num_items = data['item_id'].max() + 1
rating_array = data['rating'].values
N = rating_array.shape[0]
embedding_dim = 20
learning_rate = 0.1 #0.05就很不错了
reg_param = 0.1
num_epochs = 40
errors = []

# 随机初始化
# user_embeddings = np.random.rand(num_users, embedding_dim) * (np.sqrt(5 / embedding_dim))
# print(user_embeddings)
# item_embeddings = np.random.rand(num_items, embedding_dim) * (np.sqrt(5 / embedding_dim))

# SVD初始化
rows = []
cols = []
ratings = []
for index, row in data.iterrows():
    user_id, item_id, rating = row
    rows.append(user_id)
    cols.append(item_id)
    ratings.append(rating)

ratings_matrix_coo = sps.coo_matrix((ratings, (rows, cols)), shape=(num_users, num_items), dtype=np.float64)
ratings_matrix_csr = ratings_matrix_coo.tocsr()

U, sigma, Vt = svds(ratings_matrix_csr, k=embedding_dim)

scaler = MinMaxScaler(feature_range=(0, 0.5))

user_embeddings = scaler.fit_transform(U.reshape(-1, 1)).flatten()
item_embeddings = scaler.fit_transform(Vt.T.reshape(-1, 1)).flatten()
print("svd初始化完成")
# SVD初始化不如随机初始化，以上


# 定义损失函数与梯度
def predict(user_id, item_id):
    return np.dot(user_embeddings[user_id], item_embeddings[item_id])


def update_embeddings(user_id, item_id, rating):
    prediction = predict(user_id, item_id)
    epsilon = rating - prediction
    user_grad = -epsilon * item_embeddings[item_id] + reg_param * user_embeddings[user_id]
    item_grad = -epsilon * user_embeddings[user_id] + reg_param * item_embeddings[item_id]

    user_embeddings[user_id] -= learning_rate * user_grad
    item_embeddings[item_id] -= learning_rate * item_grad


for epoch in range(num_epochs):

    idx = 0
    evaluated_ratings = np.empty(N, dtype=np.float64)

    for index, row in data.iterrows():
        user_id, item_id, _ = row
        evaluated_ratings[idx] = predict(user_id, item_id)
        idx += 1

    E = (np.sum(np.square(rating_array - evaluated_ratings)) +
         reg_param * (np.sum(np.square(user_embeddings)) + np.sum(np.square(item_embeddings))))
    errors.append(E)

    for index, row in data.iterrows():
        user_id, item_id, rating = row
        # print(user_id, item_id, rating)
        update_embeddings(user_id, item_id, rating)

    if epoch % 10 == 0:
        print(f'errors = {E}')


epochs = list(range(0, 40))
# 创建图形
plt.figure(figsize=(10, 5))

# 绘制误差随 epoch 变化的图
plt.plot(epochs, errors, marker='o')

# 添加标题和轴标签
plt.title('Error vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')

# 显示网格
plt.grid(True)

# 显示图形
plt.show()