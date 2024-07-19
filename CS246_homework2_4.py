import numpy as np
import torch
device = 'cuda'
# 读取数据

shows = []
with open('data/shows.txt', 'r') as file:
    for line in file:
        show_name = line.strip()
        shows.append(show_name)

data = []
with open('data/user-shows.txt', 'r') as file:
    for line in file:
        row = list(map(int, line.strip().split()))
        data.append(row)


R = torch.from_numpy(np.array(data)).cuda().float()
P_inv_sqrt = torch.diag(R.sum(axis=1).pow(-0.5)).cuda().float()
Q_inv_sqrt = torch.diag(R.sum(axis=0).pow(-0.5)).cuda().float()

users_sim_S = torch.matmul(P_inv_sqrt @ torch.matmul(R, R.T), P_inv_sqrt)

# user-user collaborative filtering
UUCF = users_sim_S @ R
result1 = UUCF[499,:100]
result1_sort = torch.sort(result1, descending=True)
rec_movies_idx = result1_sort.indices[:5]
print(f"values of user-user collaborative filtering :{result1_sort.values[:5]}")
rec_movie_names = [shows[idx] for idx in rec_movies_idx.cpu().tolist()]
print(f"result of user-user collaborative filtering :{rec_movie_names}")

# user-user collaborative filtering
users_sim_S = torch.matmul(Q_inv_sqrt @ torch.matmul(R.T, R),Q_inv_sqrt)
IICF = R @ users_sim_S
result2 = IICF[499,:100]
result2_sort = torch.sort(result2, descending=True)
rec_movies_idx = result2_sort.indices[:5]
print(f"values of item-item collaborative filtering :{result2_sort.values[:5]}")
rec_movie_names = [shows[idx] for idx in rec_movies_idx.cpu().tolist()]
print(f"result of item-item collaborative filtering :{rec_movie_names}")