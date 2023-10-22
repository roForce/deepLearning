# -*- coding: utf-8 -*-            
# @Author : Code_nickusual
# @Time : 2023/10/22 16:39

'''
本模块主要是针对k近邻算法进行一个基础demo的实现，并借此完成对于机器学习算法的学习深入的巩固
    机器学习的主要的步骤：
        1.获取数据集
        2.数据的基本处理
        3.特征工程
        4.机器学习
        5.模型评估
'''
from  sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from  sklearn.neighbors import KNeighborsClassifier
import  matplotlib.pyplot  as plt
#1.获取一个数据集
dataset = load_iris()

#2.数据基本处理
## 2.1数据分割，将数据分割成训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(dataset.data,dataset.target,random_state=22,test_size=0.2)

#3.特征工程
##3.1实例化一个转换器
transfer  = StandardScaler()
##3.2调用fit_transform方法
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

#4.机器学习（模型训练）
##4.1实例化一个估计器
estimator = KNeighborsClassifier(n_neighbors=5)
##4.2模型训练
estimator.fit(x_train,y_train)

#5.模型评估
##5.1输出预测值
y_pre = estimator.predict(x_test)
print("原始值是：\n",y_test)
print("预测值是：\n",y_pre)

#6.输出准确率
#传入的值应该是x_test，和y_test
ret = estimator.score(x_test,y_test)
print("准确率是:",ret)

print(type(y_test))
#7.图像的方式对比
##7.1准备数据
x = range(y_test.size)
##7.2创建画布
plt.figure(figsize=(20,8),dpi=100)
##7.3绘制图像
plt.scatter(x,y_test,s=100,marker='*',c='r')
plt.scatter(x,y_pre,s=50,marker='*',c='b')
##7.4修改刻度
x_label = ['第{}个'.format(i) for i in x]
y_label = range(3)
plt.xticks(x,x_label)
plt.yticks(y_label)
##7.5图像显示
plt.show()











