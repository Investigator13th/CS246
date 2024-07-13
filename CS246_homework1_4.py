# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla
# Modified: Alex Porter
import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt


# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return np.sum(np.abs(u - v))


# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
#
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')


# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))

    return f


# k=r  L=b
# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low=0,
                                       high=num_dimensions,
                                       size=k)
        thresholds = np.random.randint(low=min_threshold,
                                       high=max_threshold + 1,
                                       size=k)

        functions.append(create_function(dimensions, thresholds))
    return functions


# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
# 这里返回的是输入向量的多个hash值
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])


# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))


# Retrieve all of the points that hash to one of the same buckets
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.

# 这里使用的是or增强
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
                            any(hashed_point == hashed_A[i]), range(len(hashed_A)))


# Sets up the LSH.  You should try to call this function as few times as
# possible, since it is expensive.
# A: The dataset in which each row is an image patch.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k=24, L=10):
    functions = create_functions(k=k, L=L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)


# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors=10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)

    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]


# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")


# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors=10):
    distances = map(lambda r: (r, l1(A[r], A[query_index])), range(len(A)))
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]
    return [t[0] for t in best_neighbors]  # TODO


# TODO: Write a function that computes the error measure
def cal_error(query_index, lsh_neighbors, linear_neighbors, A):
    total_l1_lsh = 0
    total_l1_linear = 0

    l1_lsh_values = [l1(A[r], A[query_index]) for r in lsh_neighbors[:3]]
    l1_linear_values = [l1(A[r], A[query_index]) for r in linear_neighbors[:3]]

    # 使用sum函数对列表求和
    total_l1_lsh += sum(l1_lsh_values)
    total_l1_linear += sum(l1_linear_values)

    denominator = max(total_l1_linear, 1e-10)
    error = total_l1_lsh / denominator

    # 考虑到整个查询集的平均错误率
    return error


# TODO: Solve Problem 4
def problem4_2(A):
    # print("读取数据中...")
    query_index_set = [i - 1 for i in range(100, 1100, 100)]
    L_list = [i for i in range(10, 22, 2)]
    # L_list = [20]
    error_list_L = []
    print('循环L中...')
    for L in L_list:
        # print("生成哈希函数族中...")
        functions, hashed_A = lsh_setup(A, L=L)
        error = 0
        for query_index in query_index_set:
            lsh_neighbors = lsh_search(A, hashed_A, functions, query_index)
            linear_neighbors = linear_search(A, query_index, len(lsh_neighbors))
            error += cal_error(query_index, lsh_neighbors, linear_neighbors, A) / (len(query_index_set))
        print(f"(L,k)=({L},24),error={error}")
        error_list_L.append(error)

    print('循环k中...')
    k_list = [i for i in range(16, 26, 2)]
    # k_list = [18]
    error_list_k = []

    for k in k_list:
        # print("生成哈希函数族中...")
        functions, hashed_A = lsh_setup(A, k=k)
        error = 0
        for query_index in query_index_set:
            lsh_neighbors = lsh_search(A, hashed_A, functions, query_index)
            linear_neighbors = linear_search(A, query_index)
            error += cal_error(query_index, lsh_neighbors, linear_neighbors, A) / (len(query_index_set))
        print(f"(L,k)=(10,{k}),error={error}")
        error_list_k.append(error)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(L_list, error_list_L)
    ax1.set_xlabel('L')
    ax1.set_ylabel('error')
    ax1.set_title('k=24')
    ax2.plot(k_list, error_list_k)
    ax2.set_xlabel('k')
    ax2.set_ylabel('error')
    ax2.set_title('L=10')
    plt.tight_layout()
    plt.show()


def problem4_1(A):
    query_index_set = [i - 1 for i in range(100, 1100, 100)]
    functions, hashed_A = lsh_setup(A)
    error = 0
    lsh_time = []
    linear_time = []
    for query_index in query_index_set:
        start = time.time()
        lsh_neighbors = lsh_search(A, hashed_A, functions, query_index)
        middle = time.time()
        lsh_time.append(middle - start)
        linear_neighbors = linear_search(A, query_index)
        end = time.time()
        linear_time.append(end - middle)

        # error += cal_error(query_index, lsh_neighbors, linear_neighbors, A) / (len(query_index_set))

    print(f"局部敏感哈希算法平均时间：{sum(lsh_time) / len(lsh_time)}\n"
          f"线性搜索平均时间：{sum(linear_time) / len(linear_time)}")

    # raise NotImplementedError


def problem4_3(A, L, k):
    query_index_set = [99]
    functions, hashed_A = lsh_setup(A, L, k)
    for query_index in query_index_set:
        lsh_neighbors = lsh_search(A, hashed_A, functions, query_index)
        linear_neighbors = linear_search(A, query_index)
        plot(A, lsh_neighbors, "results\lsh")
        plot(A, linear_neighbors, "results\linear")


#### TESTS #####

# class TestLSH(unittest.TestCase):
#     def test_l1(self):
#         u = np.array([1, 2, 3, 4])
#         v = np.array([2, 3, 2, 3])
#         self.assertEqual(l1(u, v), 4)
#
#     def test_hash_data(self):
#         f1 = lambda v: sum(v)
#         f2 = lambda v: sum([x * x for x in v])
#         A = np.array([[1, 2, 3], [4, 5, 6]])
#         self.assertEqual(f1(A[0, :]), 6)
#         self.assertEqual(f2(A[0, :]), 14)
#
#         functions = [f1, f2]
#         self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
#         self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))
# #
# #     ### TODO: Write your tests here (they won't be graded,
# #     ### but you may find them helpful)


if __name__ == '__main__':
    # unittest.main() ### TODO: Uncomment this to run tests
    A = load_data("data/patches.csv")
    # problem4_1(A)
    # 局部敏感哈希算法平均时间：0.08982794284820557
    # 线性搜索平均时间：0.3357378959655762
    problem4_2(A)
    # 取L=20,k=28
    # L = 20
    # k = 28
    # problem4_3(A, L, k)
