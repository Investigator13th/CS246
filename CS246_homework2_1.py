import numpy as np
import scipy.linalg
from scipy.linalg import svd

# 创建一个矩阵
M = np.array([[1, 2],
              [2, 1],
              [3, 4],
              [4, 3]])

# 进行SVD分解
U, s, VT = svd(M)

# 输出分解后的矩阵
print("U:\n", U)
print("s (singular values):\n", s)
print("V\n", VT.T)

# 计算M^T*M的特征分解
evals, evecs = scipy.linalg.eigh(M.T@M)

index_value_pairs = [(i, val) for i, val in enumerate(evals)]

sorted_pairs = sorted(index_value_pairs, key=lambda x: x[1], reverse=True)

sorted_values = [val for _, val in sorted_pairs]
sorted_vectors = evecs[:, [i for i, _ in sorted_pairs]]
print("特征值:\n", sorted_values)
print("特征向量:\n", sorted_vectors)

print(VT.T == sorted_vectors)

# 看起来是绝对值相同