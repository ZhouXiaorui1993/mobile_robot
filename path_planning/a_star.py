#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""用Python实现简单的A*算法

A*算法简介：

- 搜索区域：二维数组，相当于栅格地图，每个栅格称为一个节点
- 开放列表（OpenList）：用于存储待检测的节点
- ClosedList：用于存储已经检测过的节点
- 父节点：在路径规划中用于回溯的节点
- 公式：F(n)=G+H，其中G是初始位置沿着已生成的路径到指定待检测栅格的移动代价，H是待检测栅格到目标节点的移动代价
- 启发函数（Heuristics Function）：H为启发函数，被认为是一种试探，由于在找到唯一的路径前，我们不确定在前面会出现什么障碍物，因此计算H的方法具体应该
由实际场景决定。这里简化为，曼哈顿距离，即横向距离和纵向距离之和
"""
import cv2
import numpy as np
from math import sqrt


# 栅格地图
class GridMap(object):
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.__map = np.zeros((self.height, self.width), dtype=np.uint8)

    def get_map(self):
        """
        获取地图数据
        :return:
        """
        return self.__map

    def add_obstacle(self, ld, ru):
        """
        为地图增加障碍物
        :param ld:左下角坐标[x,y]
        :param ru: 右上角坐标[x,y]
        :return: 0——添加成功，1——添加失败
        """
        try:
            self.__map[ld[1]:ru[1], ld[0]:ru[0]] = 100
            return 0
        except IndexError:
            print("障碍物超出地图范围")
            return 1

    def change_grid_value(self, grid_pos, value):
        """
        改变地图某个栅格的值
        :param grid_pos: 坐标，list形式，[x,y]
        :param value: 0-255
        :return: 改变成功——0，失败——1
        """
        if value < 0 or value > 255:
            return 1
        else:
            self.__map[grid_pos[1], grid_pos[0]] = value


class MapNode(object):
    def __init__(self, x, y):
        # 坐标
        self.x_pos = x
        self.y_pos = y

        self.node_id = None

        # 父节点
        self.parent_node = None
        # 子节点
        self.children_node = None

        # G\H\F
        self.g_value = None
        self.h_value = None
        self.f_value = None

    def get_node_pos(self):
        """
        获取节点的坐标
        :return:
        """
        return [self.x_pos, self.y_pos]

    def get_parent_node(self):
        """
        获取父节点
        :return: 该节点的父节点
        """
        if self.parent_node is not None:
            return self.parent_node
        else:
            return "no parent_node"


def check_node_valid(node, map):
    """检查输入的node是否在地图上"""
    # 检查输入的起始点和目标点是否可行

    if 0 <= node.x_pos < map.width and 0 <= node.y_pos < map.height:
        return True
    else:
        print("输入的node坐标 %s 不合法" % [node.x_pos, node.y_pos])
        return False


def calc_g(node1, node2):
    """
    计算两点间的G值
    :param node1: 被检测的点1
    :param node2: 相邻点2
    :param grid_map: 栅格地图
    :return: G（两点间的距离）
    """
    g_value = sqrt((node1.x_pos - node2.x_pos)**2 + (node1.y_pos - node2.y_pos)**2)

    return g_value


def calc_h(node, goal_node):
    """
    启发函数，这里用的是曼哈顿距离
    :param node: open list中的节点
    :param goal_node: 目标节点
    :param grid_map: 栅格地图
    :return: h_value 两栅格之间的曼哈顿距离
    """
    h_value = abs(node.x_pos - goal_node.x_pos) + abs(node.y_pos - goal_node.y_pos)

    return h_value


def judge_in_openlist(node, openlist):
    """判断某个节点是否位于某个list"""
    for i in openlist:
        if node.x_pos == i.x_pos and node.y_pos == i.y_pos:
            return True
    return False


def a_star(test_map, start_node=MapNode(0,0), goal_node=MapNode(90,90)):
    """
    返回组成最优路径的节点list
    :param test_map: 测试地图
    :param start_node: 起始节点
    :param goal_node: 目标节点
    :return:
    """

    # 获取地图数据
    test_map_data = test_map.get_map()

    # 用于存储已经检测过的节点
    closed_list = set()
    # 用于存储还未检测的节点
    open_list = set()

    proc_node = start_node

    # 检查要处理的node是否合法
    if check_node_valid(proc_node, test_map) and check_node_valid(goal_node, test_map):
        # 将起始点插入open list
        open_list.add(proc_node)

    i = 0

    while True:
        print i
        i = i+1

        # 得到该点周围的可行节点
        dirc_list = [[0, 1], [1, 1], [1, 0], [-1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

        for dirc in dirc_list:
            node = MapNode(proc_node.x_pos+dirc[0], proc_node.y_pos+dirc[1])

            # 如果节点在地图上，且不在障碍物内，且不在close list中， 则添加到open_list
            if check_node_valid(node, test_map) and test_map_data[node.y_pos, node.x_pos] == 0 and not\
                    judge_in_openlist(node, closed_list):
                # 检查该节点是否在Openlist中，如果在，则检查是否该节点和当前处理节点之间的g值是否小于原g值
                if judge_in_openlist(node, open_list):
                    # 计算当前g值
                    now_g_value = calc_g(node, proc_node)
                    # 若当前g值更小，证明从当前处理节点走代价更小，则将当前处理节点设为该节点父节点
                    if now_g_value < node.g_value:
                        node.parent_node = proc_node
                    else:
                        continue
                # 如果节点不在openlist中
                else:
                    # 则加入该节点，并设置当前处理节点为其父节点
                    node.parent_node = proc_node
                    open_list.add(node)

        # 将原节点从open list中移除，添加到close list
        open_list.remove(proc_node)
        closed_list.add(proc_node)

        # 在openlist中选取F值最小的的节点作为新的待检测节点
        # F = G + H
        f_min = 100000  # 初始化，用于存储和第一次比较

        for node in open_list:
            node.g_value = calc_g(proc_node, node)
            node.h_value = calc_h(node, goal_node)
            node.f_value = node.g_value + node.h_value
            if node.f_value < f_min:
                f_min = node.f_value
                proc_node = node

        # 判断此时目标节点是否位于openlist中，如果是，则设置goal_node的父节点为当前处理节点，退出循环
        if judge_in_openlist(goal_node, open_list):
            goal_node.parent_node = proc_node
            break

    # 从终点开始，沿着它的父节点向上回溯，得到最终路径
    final_path = list()
    iter_node = goal_node

    while iter_node != start_node:
        final_path.append(iter_node)
        iter_node = iter_node.parent_node
    final_path.append(start_node)

    # for i in range(len(final_path)-1, -1, -1):
    #     node = final_path[i]
    #     print ("(%s, %s) -> " % (node.x_pos, node.y_pos)),

    # 路径为白色
    for node in final_path:
        grid = [node.x_pos, node.y_pos]
        test_map.change_grid_value(grid, 255)

    # 用openCV画出地图和路径
    cv2.imshow("test_map", test_map_data)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # 创建地图
    test_map = GridMap(300, 400, 1)

    # 添加障碍物
    test_map.add_obstacle([23, 23], [60, 40])

    test_map.add_obstacle([30, 50], [120, 60])

    test_map.add_obstacle([100, 50], [120, 200])

    test_map.add_obstacle([100, 200], [200, 220])

    # 设置起始节点
    start_node = MapNode(2, 2)
    goal_node = MapNode(250, 300)

    a_star(test_map, start_node, goal_node)
