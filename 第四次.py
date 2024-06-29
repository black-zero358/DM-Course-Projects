import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

import warnings
warnings.filterwarnings('ignore')

import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


data = pd.read_excel('超市营业额.xlsx')

df1 = copy.deepcopy(data)
print(df1.tail())

# 预处理: (一) 重复值
print('*总行数=', len(df1))
print('*重复行数=', len(df1[df1.duplicated()]))
print('*重复的行记录: \n', df1[df1.duplicated()])
# 完全重复
df1.drop_duplicates(inplace=True)
print('*去除重复行之后的总行数=', len(df1))
# 部分重复
# 打印数据中的姓名与工号
df2 = df1[['姓名', '工号']]
df2.drop_duplicates()

# 预处理: (二) 交易额的异常值
# 异常值1:低于200的交易额，用200替换
# 异常值2:高于3000的交易额，用3000替换
# 查看异常值1
df1['交易额'].astype(float)
print(df1[df1['交易额'] < 200])
# 处理异常值1:用loc[行索引，列索引]定位
df1.loc[df1['交易额'] < 200, '交易额'] = 200
print(df1[df1['交易额'] == 200])
# 查看异常值2
df1['交易额'].astype(float)
print(df1[df1['交易额'] > 3000])
# 处理异常值2
df1.loc[df1['交易额'] > 3000, '交易额'] = 3000
print(df1[df1['交易额'] == 3000])
# 检查异常值是否全部处理
df1.loc[(df1['交易额'] < 200) | (df1['交易额'] > 3000), '交易额'].count()

# 预处理: (三)缺失值
# 查看缺失值: .isna()或者.isnull()
print(df1.loc[df1['交易额'].isnull()])
# 处理缺失值:丢弃
print('处理前总行数=', len(df1))
df1.dropna(inplace=True)
print('处理后总行数=', len(df1))
# 处理缺失值:替换
# 固定值替换1000
df1.loc[df1['交易额'].isnull(), '交易额'] = 1000
print(df1.iloc[[109, 123, 167], :])  # 查看被替换的行
# 并对有缺失值的行，用每个人自己的平均交易额替换
rows = df1[df1["交易额"].isnull()]
print(rows, '\n')
for i in rows.index:
    imean = round(df1.loc[df1['姓名'] == df1.loc[i, '姓名'], '交易额'].mean(), 2)
    df1.loc[i, '交易额'] = imean
    print(i, ', imean')
print(df1.iloc[[109, 123, 167], :])  # 查看被替换的行

# 分组聚合:按姓名分组
df_grouped1 = df1.groupby('姓名')
# 其中某名营业员的记录，显示前5行
df_grouped1.get_group('张三').head()
# 每个人的平均营业额
round(df_grouped1['交易额'].mean(), 2)

# 按时段分组
df_grouped2 = df1.groupby('时段')
# 计算每个时段的统计值
df_grouped2['交易额'].agg([np.sum, np.max, np.min, np.mean, np.std]).round(2)

# 透视表:观察每人在不同时段的交易额(平均)
df_ptable = df1.pivot_table(index='姓名', columns='时段', values='交易额')
print(df_ptable)

# 绘图:每个人在不同时段的交易额对比
plt.figure(figsize=(8, 5))
df_ptable.plot(kind='bar')
plt.legend(bbox_to_anchor=(1, 0.8))

# 透视表:观察每人在不同柜台的交易额(平均)
df_ptable = df1.pivot_table(index='姓名', columns='柜台', values='交易额')
print(df_ptable)

# 绘图:每个人在不同柜台的交易额对比
df_ptable.plot(kind='bar', legend=None, figsize=(10, 6))
plt.legend(loc='upper left')
plt.show()

# 透视表:观察每人每天的交易额
df_ptable = df1.pivot_table(index='姓名', columns='日期', values='交易额')
print(df_ptable)

# 绘图:每个人每天的交易额对比
plt.figure(figsize=(12, 6))
df_ptable.plot(kind='bar', legend=None, figsize=(10, 6))
plt.show()
