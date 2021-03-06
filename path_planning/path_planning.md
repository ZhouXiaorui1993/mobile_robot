# 路径规划简介

## 路径规划简介

所谓路径规划，其实是依据某种最优准则，在工作空间中寻找一条从起始状态到目标状态、且可以避开障碍物的最优路径。

**路径规划需要解决的问题**

1. 始于初始点，止于目标点

2. 避障

3. 尽可能的优化路径


## 路径规划的分类

- 静态结构化环境下的路径规划
- 动态已知环境下的路径规划
- 动态不确定环境下的路径规划

## 路径规划算法的分类

### 分类一

- 基于采样的方法
    - Voronol
    - RRT
    - PRM

- 基于节点的方法
    - Dijkstra(迪杰斯特拉)
    - A*
    - D*

- 基于数学模型的方法
    - MILP
    - NLP

- 基于生物启发式的算法
    - NN
    - GN

- 多融合算法
    - PRM-Node
    - based

### 分类二

- 传统算法
    - 模拟退化法
    - 人工势场法
    - 等等

- 智能算法
    - 遗传算法
    - 神经网络算法
    - 等等

## A*算法

### 算法简介

- 基于栅格地图，栅格用于划分空间，简化搜索区域

- 栅格（二维数组中的元素）被标记为可行或不可行，未来的路径就是一系列可行栅格的集合。

- 其中有个Node（节点）的概念，它表示栅格

- 算法的大体思路是从初始节点（起始位置）开始，搜索其附近的节点，直到找到目标点，连接一系列可行节点，即得到期望的路径

- 具体步骤如下:
    - step1：给每个节点赋值 F = G+H，其中G表示从初始点到给定待查节点的距离（距离可以有多种度量方法）；H表示从给定待查节点到目标点B的距离（Heuristic计算距离时忽略到达目标点会遇到的障碍）
    - step2：找到F值最小的节点作为新的起点，将它从OpenList中删除，加入到ClosedList中。然后检查它的临近节点，忽略已经在ClosedList中的节点和不可行节点（障碍）。如果临近节点已经在OpenList里面，则对比一下是否从现节点到临近节点的G值比原G值小，若是，则把现节点作为父节点；若否，则不做改动
    - step3：如果上步骤中的新节点未造成任何改动，则继续在OpenList中寻找新的节点
    - step4：重复步骤1-3，直到找到目标节点

### 算法特点

- A*算法在理论上是时间最优的，但其缺点是空间增长是指数级的

- **关于`D*`算法**

    它是动态的`A*`算法（Dynamic `A*`），应用于在动态环境下的路径搜索

- `A*`算法是**广度优先**的，即先搜索不同分支同一层次的成员，然后再搜索所有分支下一个层次的成员，直到搜索完毕（还有一个概念是深度优先，即先搜索一个分支的所有成员，然后再搜索其他分支的搜索顺序）
    
    
    
    
    
    
    
     
    
    
    
    
