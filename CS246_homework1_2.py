import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 读数据
file_path = 'data/browsing.txt'
with open(file_path, 'r') as file:
    data = file.readlines()

# 将数据集转换为列表
transactions = [basket.split() for basket in data]

# 使用TransactionEncoder将数据转换为适合的格式
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 计算频繁项集
print('计算频繁项集中...')
frequent_itemsets = apriori(df, min_support=100/len(df), use_colnames=True, max_len=3)

# 过滤出项集大小为1的频繁项集
size_1_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 1)]

# 输出项集大小为1的频繁项集的个数
print("大小为1的频繁项集的个数为:", len(size_1_itemsets))

# 计算关联规则
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0)

# 按置信度降序排列规则
rules = rules.sort_values(by="confidence", ascending=False)

# 分别提取 X→Y 和 (X,Y)→Z 规则
rules_X_to_Y = rules[rules['antecedents'].apply(lambda x: len(x) == 1) & rules['consequents'].apply(lambda x: len(x) == 1)]
rules_XY_to_Z = rules[rules['antecedents'].apply(lambda x: len(x) == 2) & rules['consequents'].apply(lambda x: len(x) == 1)]

# 输出结果
print("频繁项集：")
print(frequent_itemsets)

print("\nX→Y 规则：")
print(rules_X_to_Y[['antecedents', 'consequents', 'confidence']])

print("\n(X,Y)→Z 规则：")
print(rules_XY_to_Z[['antecedents', 'consequents', 'confidence']])
