#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""zxr 2018/10/6
基于自适应控制的思想设计的路径跟踪控制器
可在运行过程中改变线速度v的大小(由键盘输入)
"""


from __future__ import division  # 使用python3的除法

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from rbx1_nav.transform_utils import quat_to_angle, normalize_angle
from math import cos, sin
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
# 用于获取输入
import tty
import sys
import termios
import threading


class PathFollowing(object):
    def __init__(self):

        # 初始化节点
        rospy.init_node("ma_path_following", anonymous=False)

        # 注册回调函数
        rospy.on_shutdown(self.shut_down)

        # 发布控制消息的句柄
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 机器人机体坐标系
        self.base_frame = rospy.get_param('~base_frame', '/base_link')

        # 里程计坐标系
        self.odom_frame = rospy.get_param('~odom_frame', '/odom')

        # 初始化一个tf监听器
        self.tf_listener = tf.TransformListener()

        # tf需要时间填满缓存
        rospy.sleep(2)

        # Find out if the robot uses /base_link or /base_footprint
        try:
            self.tf_listener.waitForTransform(self.odom_frame, '/base_footprint', rospy.Time(), rospy.Duration(1))
            self.base_frame = '/base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, '/base_link', rospy.Time(), rospy.Duration(1))
                self.base_frame = '/base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between /odom and /base_link or /base_footprint")
                rospy.signal_shutdown("tf Exception")

        # 期望路径
        self.desire_path = 'x**2 + y**2 -4 = 0'

        # 初始状态
        self.s = 0
        theta_es0 = 0.1
        self.p1_es = sin(theta_es0)
        self.p2_es = cos(theta_es0)

        # 机器人车前一点的长度
        self.L = 0.1  # 误差变化和L有关，L越大，收敛越快

        # 机器人的线速度设为常数
        self.v = 0.3

        # 键盘输入的设置
        self.in_ch = None # 用于保存输入的字符
        self.k_v = {'d': 0.5, 'u': 1.6}

        # 最大线速度和角速度
        self.max_angular_vel = 1
        self.max_linear_vel = 0.7

        # 设置仿真时间
        self.sim_time = 100

        # 仿真步长
        self.dT = 0.01

        # 控制参数
        self.k = 0.5
        self.beta = 1

        # 存储机器人的位置信息
        self.x_list = list()
        self.y_list = list()
        self.theta_list = list()

        # 期望路径信息
        self.xd_list = list()
        self.yd_list = list()

        # 保存速度和角速度
        self.v_list = list()
        self.w_list = list()

        # 保存误差信息的list
        self.ex_list = []
        self.ey_list = []

        # 保存仿真时间信息
        self.sim_time_list = []

        # 记录当前时间
        sim_start_time = rospy.Time.now().secs

        # How fast will we check the odometry values?
        rate = 10  # 发布频率

        # Set the equivalent ROS rate variable
        r = rospy.Rate(rate)

        # 开始等待输入的子线程
        wait_for_input = threading.Thread(target=self.get_key)
        wait_for_input.start()

        # 循环
        while rospy.Time.now().secs - sim_start_time < self.sim_time and not rospy.is_shutdown():

            if self.in_ch is not None:
                if self.in_ch in self.k_v:
                    self.v = self.v * self.k_v[self.in_ch]
                    self.in_ch = None  # 重置为None
                    print("after change: v = %s\n" % self.v)
                elif ord(self.in_ch) == 0x3:  # ctrl-c 退出循环
                    break

            # 得到机器人当前位姿
            (position, rotation) = self.get_odom()

            # 位姿信息
            self.x = position.x
            self.y = position.y
            self.theta = quat_to_angle(rotation)

            # 存入list
            self.x_list.append(self.x+self.L*cos(self.theta))
            self.y_list.append(self.y+self.L*sin(self.theta))
            # self.x_list.append(self.x)
            # self.y_list.append(self.y)
            self.theta_list.append(self.theta)

            # 期望路径
            h1 = 1.5
            h2 = 1
            self.x_d = h1*cos(h2*self.s)
            self.y_d = h1*sin(h2*self.s)
            # 对s求偏导数
            dx_d = -h1*h2*sin(h2*self.s)
            dy_d = h1*h2*cos(h2*self.s)

            # 存入list
            self.xd_list.append(self.x_d)
            self.yd_list.append(self.y_d)

            # 误差
            self.ex = self.x + self.L*cos(self.theta) - self.x_d
            self.ey = self.y + self.L*sin(self.theta) - self.y_d

            # 保存到list
            self.ex_list.append(self.ex)
            self.ey_list.append(self.ey)

            # 计算控制输入
            # 系数矩阵
            Q_es = np.array([[-self.L * self.p1_es, -dx_d], [self.L*self.p2_es, -dy_d]])

            B = np.array([[-self.v*self.p2_es - self.k * self.ex], [-self.v*self.p1_es - self.k * self.ey]])
            # 控制矩阵
            U = np.dot(inv(Q_es), B)  # NumPy中的乘法运算符*指示按元素计算，矩阵乘法可以使用dot函数或创建矩阵对象实现
            # 控制输入
            self.w = U[0, 0]  # U是一个矩阵，必须这样取值，不然U[0]取得的是一个list
            self.ds = U[1, 0]

            # 更新路径变量和自适应控制律
            self.adaptive_controller(self.dT, self.beta)

            # 室内机器人，对控制器进行限幅
            self.w = self.limit_velocity(self.w, self.max_angular_vel)
            self.v = self.limit_velocity(self.v, self.max_linear_vel)
            # 保存到list
            self.v_list.append(self.v)
            self.w_list.append(self.w)

            # 保存仿真时间到list
            self.sim_time_list.append(rospy.Time.now().secs - sim_start_time)

            # 初始化一个空的运动控制消息
            self.cmd_vel = Twist()

            # 填充
            self.cmd_vel.angular.z = self.w
            self.cmd_vel.linear.x = self.v

            self.cmd_vel_pub.publish(self.cmd_vel)

            # r.sleep()

            rospy.sleep(self.dT)

        # 循环结束，停下机器人
        rospy.loginfo("循环结束，停下机器人")
        self.cmd_vel_pub.publish(Twist())
        # self.plot_figures()

    @staticmethod  # 静态方法，与类属性无关
    def limit_velocity(x, max_value):
        # 限幅函数
        if abs(x) > max_value:
            if x < 0:
                return -max_value
            else:
                return max_value
        else:
            return x

    def adaptive_controller(self, dt, beta=1.5):
        """更新自适应控制律"""
        # beta = 1.5
        self.s = self.s + self.ds * dt

        dp1_es = self.beta * (
                -self.L * self.ex * self.w + self.v * self.ey) + self.w * self.p2_es
        dp2_es = self.beta * (
                self.L * self.ey * self.w + self.v * self.ex) - self.w * self.p1_es

        p1_es_next = self.p1_es + dp1_es * self.dT
        p2_es_next = self.p2_es + dp2_es * self.dT
        self.p1_es = p1_es_next
        self.p2_es = p2_es_next



    def get_odom(self):
        # Get the current transform between the odom and base frames
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return Point(*trans), Quaternion(*rot)

    def get_key(self):
        while True:
            print("waiting for input from keyboard...\n"
                  "d: v=v*0.5, u: v=v*2")
            tty_fd = sys.stdin.fileno()  # 获取标准输入的文件描述符
            tty_old_settings = termios.tcgetattr(tty_fd)
            try:
                tty.setraw(tty_fd)
                self.in_ch = sys.stdin.read(1)  # 每次读取一个字符进行处理
                if ord(self.in_ch) == 0x3:  # ctrl-c 退出循环
                    break
            finally:
                termios.tcsetattr(tty_fd, termios.TCSADRAIN, tty_old_settings)  # 还原终端设置
            # return ch

    def plot_figures(self):
        # 运行轨迹
        plt.figure(1)
        plt.plot(self.xd_list, self.yd_list, linestyle='--', linewidth=2, color='blue', label='desired path')
        plt.plot(self.x_list, self.y_list, linestyle='--', linewidth=2, color='red', label='real path')
        plt.title('The tracking trajectory')
        plt.xlabel('x/m', fontsize=16)
        plt.ylabel('y/m', fontsize=16)
        plt.legend()
        plt.grid(True)  # 打开网格

        # 误差
        plt.figure(2)
        plt.title('position tracking error')
        # times = [i for i in range(self.loop_count)]
        # label_error = ['ex', 'ey']
        plt.plot(self.sim_time_list, self.ex_list, 'r-', label='ex')
        plt.plot(self.sim_time_list, self.ey_list, 'g-', label='ey')
        plt.legend()

        # 速度和角速度
        plt.figure(3)
        plt.title('the generated control signals')
        plt.plot(self.sim_time_list, self.v_list, 'r', label='linear velocity')
        plt.plot(self.sim_time_list, self.w_list, 'g', label='angular velocity')
        # 设置坐标轴名称
        plt.xlabel('t/s')
        plt.ylabel('v/(m/s); w/(rad/s)')
        # plt.ylim((-0.5, 0.3))
        plt.legend()  # 可以写作legend(loc=0, ncol=1), loc设置显示的位置，0表示自适应；ncol设置显示的列数,1为两列

        # 显示图像
        plt.show()

    def shut_down(self):
        """停机回调函数"""

        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
        # 循环结束，画图
        self.plot_figures()


if __name__ == '__main__':


    PathFollowing()
    # try:
    #     PathFollowing()
    # except Exception as e:
    #     print(e)
    #     rospy.loginfo("path_following node terminated.")