#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" nav_square.py - Version 1.0 2018-4-20

    假设theta角不可测量，自适应＋反馈线性化控制方法验证。
    大循环（计算控制输入），间隔调用【生成参考轨迹、自适应控制律】的函数。
    利用python定时器实现的间隔循环调用。
      
"""

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from rbx1_nav.transform_utils import quat_to_angle, normalize_angle
from math import radians, copysign, sqrt, pow, pi, cos, sin
import numpy as np
from numpy.linalg import inv
import threading
import matplotlib.pyplot as plt
import json


class NavSquare(object):
    def __init__(self):

        # 参考轨迹的一些参数
        self.get_init_value()
        # 最大速度设定
        self.max_v = 0.4
        self.max_w = 0.8
        # 参考轨迹的一些参数
        # 参考轨迹位姿初值
        self.x_d = self.x_d0 + 0.4
        self.y_d = self.y_d0 + 0.4
        self.theta_d = self.theta_d0

        # 角度估计初值
        self.theta_es0 = self.theta_d+0.2

        # 速度和角速度
        self.vd = 0.1
        self.wd = 0.05

        # cos(theta)和sin(theta)的估计初值
        self.p1_es = sin(self.theta_es0)
        self.p2_es = cos(self.theta_es0)

        # 保存参考轨迹位置信息
        self.x_d_list = []
        self.y_d_list = []

        # 保存当前机器人位置信息
        self.x_list = []
        self.y_list = []

        # 保存误差信息的list
        self.ex_list = []
        self.ey_list = []

        # 保存速度和角速度信息
        self.v_list = []
        self.w_list = []

        # 保存仿真时间信息
        self.sim_time_list = []

        # 保存估计误差信息
        self.p1_es_error_list = []
        self.p2_es_error_list = []

        # time
        self.dT = 0.1
        self.sim_time = 300

        # 控制参数
        self.k = 0.3  # 反馈控制律
        self.beta = 1  # 自适应控制律
        self.L = 0.2  # 车前一点
        
        # 循环次数
        self.loop_count = 0

        # Give the node a name
        rospy.init_node('nav_square', anonymous=False)
        
        # Set rospy to execute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)

        # How fast will we check the odometry values?
        rate = 10  # 发布频率
        
        # Set the equivalent ROS rate variable
        r = rospy.Rate(rate)
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
         
        # The base frame is base_footprint for the TurtleBot but base_link for Pi Robot
        self.base_frame = rospy.get_param('~base_frame', '/base_link')

        # The odom frame is usually just /odom
        self.odom_frame = rospy.get_param('~odom_frame', '/odom')

        # Initialize the tf listener
        self.tf_listener = tf.TransformListener()
        
        # Give tf some time to fill its buffer
        rospy.sleep(2)
        
        # Set the odom frame
        self.odom_frame = '/odom'
        
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

        # Initialize the position variable as a Point type
        # position = Point()

        # mark now time
        start_time = rospy.Time.now().secs
        # print('time.time() = %s; rospy.Time.now() = %s; rospy.Time.now().secs = %s' %
        # (start_time, rospy.Time.now(), rospy.Time.now().secs))

        while rospy.Time.now().secs - start_time < self.sim_time and not rospy.is_shutdown():
            # 休息一下
            r.sleep()
            # 得到当前位姿信息
            (position, rotation) = self.get_odom()
            # 位置测量信息
            self.x = position.x
            self.y = position.y

            # 角度的测量信息
            self.theta = rotation

            # 位置误差
            self.ex = self.x + self.L * cos(self.theta) - self.x_d
            self.ey = self.y + self.L * sin(self.theta) - self.y_d
            
            # 保存到list
            self.ex_list.append(self.ex)
            self.ey_list.append(self.ey)

            # # 将当前位置保存到list
            # self.x_list.append(self.x)
            # self.y_list.append(self.y)

            # 将当前位置(车前一点)保存到list
            self.x_list.append(self.x + self.L * cos(self.theta))
            self.y_list.append(self.y + self.L * sin(self.theta))

            # 将估计误差保存到list
            self.p1_es_error_list.append(sin(self.theta)-self.p1_es)
            self.p2_es_error_list.append(cos(self.theta)-self.p2_es)

            # 检查是否满足初始条件
            # print('%s: x = %s, y = %s, theta = %s\n' % (self.loop_count, self.x, self.y, self.theta))
            p1, p2 = sin(self.theta), cos(self.theta)
            if self.loop_count == 0 and self.beta * (self.ex ** 2 + self.ey ** 2) + (p1 - self.p1_es) ** 2 + (
                    p2 - self.p2_es) ** 2 > 1:
                print("不满足初始条件，请检查后重新设定\n")
                break

            # 计算控制输入
            # 系数矩阵
            A_es = np.array([[self.p2_es, -self.p1_es], [self.p1_es, self.p2_es]])

            dx_d = self.vd * cos(self.theta_d)
            dy_d = self.vd * sin(self.theta_d)
            B = np.array([[dx_d - self.k * self.ex], [dy_d - self.k * self.ey]])
            # 控制矩阵
            U = np.dot(inv(A_es), B)  # NumPy中的乘法运算符*指示按元素计算，矩阵乘法可以使用dot函数或创建矩阵对象实现
            # 控制输入
            self.v = U[0][0]  # U是一个矩阵，必须这样取值，不然U[0]取得的是一个list
            self.w = (1 / self.L) * U[1][0]

            # 室内机器人，限速
            self.v = self.limit_velocity(self.v, self.max_v)
            self.w = self.limit_velocity(self.w, self.max_w)

            # 保存到list
            self.v_list.append(self.v)
            self.w_list.append(self.w)

            # Initialize the movement command
            move_cmd = Twist()

            # Set the movement command to forward motion
            move_cmd.linear.x = self.v
            # Set the movement command to a rotation
            move_cmd.angular.z = self.w

            # 发布运动命令
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(0.01)
            # r.sleep()

            # 生成参考轨迹和自适应控制律
            timer = threading.Timer(0.09, self.desired_signals)
            timer.start()
            # print('%s: x_d = %s, y_d = %s\n' % (self.loop_count, self.x_d, self.y_d))

            # 循环计数
            self.loop_count = self.loop_count + 1

            # 保存仿真时间到list

            self.sim_time_list.append(rospy.Time.now().secs - start_time)
            # 显示仿真时间
            print(rospy.Time.now().secs - start_time)

        # Stop the robot when we are done
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())

        # 绘图
        self.plot_figures()

    def desired_signals(self):
        # 生成离散化的时变轨迹

        # 保存到list
        self.x_d_list.append(self.x_d)
        self.y_d_list.append(self.y_d)

        # circle1
        x_d_next = self.x_d + self.vd * cos(self.theta_d) * self.dT
        y_d_next = self.y_d + self.vd * sin(self.theta_d) * self.dT
        theta_d_next = self.theta_d + self.wd * self.dT

        # 自适应控制律
        gx_k = (1 / (self.p1_es ** 2 + self.p2_es ** 2)) * (self.vd * cos(self.theta_d) - self.k * self.ex)
        gy_k = (1 / (self.p1_es ** 2 + self.p2_es ** 2)) * (self.vd * sin(self.theta_d) - self.k * self.ey)

        p1_es_next = self.p1_es + (self.beta * (self.ex * (self.p1_es * gx_k - self.p2_es * gy_k) + self.ey *
                                                (self.p2_es * gx_k + self.p1_es * gy_k)) + self.w * self.p2_es)*self.dT
        p2_es_next = self.p2_es + (self.beta * (self.ex * (self.p2_es * gx_k + self.p1_es * gy_k) + self.ey *
                                                (self.p2_es * gy_k - self.p1_es * gx_k)) - self.w * self.p1_es)*self.dT

        # 将计算出的下一次循环所需的值赋给相应变量
        self.x_d = x_d_next
        self.y_d = y_d_next
        self.theta_d = theta_d_next
        self.p1_es = p1_es_next
        self.p2_es = p2_es_next

    def plot_figures(self):
        """绘制图像"""
        # # 误差函数
        # self.V_tr_list = []
        # self.V_es_list = []
        # for i in range(len(self.sim_time_list)):
        #     self.V_tr_list.append(sqrt(self.ex_list[i] ** 2 + self.ey_list[i] ** 2))
        #     self.V_es_list.append(sqrt(self.p1_es_error_list[i]**2 + self.p2_es_error_list[i]**2))

        # 参考轨迹
        plt.figure(1)
        plt.title('the tracking trajectory')
        plt.plot(self.x_d_list, self.y_d_list, 'r-', label='desired trajectory')
        plt.plot(self.x_list, self.y_list, 'g-', label='tracking trajectory')
        # 设置坐标轴名称
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.legend()  # 不加这一句不会正确显示图例

        # 位置跟踪误差
        plt.figure(2)
        plt.title('the position tracking error')
        plt.plot(self.sim_time_list, self.ex_list, 'r-', label='ex')
        plt.plot(self.sim_time_list, self.ey_list, 'g-', label='ey')
        # plt.plot(self.sim_time_list, self.V_tr_list)
        # 设置坐标轴范围
        plt.ylim((-1, 1))
        # 设置坐标轴名称
        plt.xlabel('t/s')
        plt.ylabel('position tracking error')
        # plt.ylabel('V_tr')
        # 设置坐标轴刻度
        plt.legend()  # 若未设置图例,则无需这句,加上会报错

        # 姿态角估计误差
        plt.figure(3)
        plt.title('the estimation error')
        plt.plot(self.sim_time_list, self.p1_es_error_list, 'r', label='p1_es_error')
        plt.plot(self.sim_time_list, self.p2_es_error_list, 'g', label='p2_es_error')
        # plt.plot(self.sim_time_list, self.V_es_list)
        plt.xlabel('t/s')
        # 设置坐标轴范围
        plt.ylim((-1, 1))
        plt.ylabel('estimation error')
        # plt.ylabel('V_es')
        plt.legend()

        # 速度和角速度
        plt.figure(4)
        plt.title('the generated control signals')
        plt.plot(self.sim_time_list, self.v_list, 'r', label='linear velocity')
        plt.plot(self.sim_time_list, self.w_list, 'g', label='angular velocity')
        # 设置坐标轴名称
        plt.xlabel('t/s')
        plt.ylabel('v/(m/s); w/(rad/s)')
        plt.ylim((-0.5, 0.3))
        plt.legend()  # 可以写作legend(loc=0, ncol=1), loc设置显示的位置，0表示自适应；ncol设置显示的列数,1为两列

        # 显示图像
        plt.show()
    
    def get_odom(self):
        # Get the current transform between the odom and base frames
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return Point(*trans), quat_to_angle(Quaternion(*rot))

    def write_final_value(self):
        # 将当前x,y,theta值写入json文件
        position_dict = dict()  # 写入dict
        position_dict['x'] = self.x
        position_dict['y'] = self.y
        position_dict['theta'] = self.theta
        # 序列化为json对象并写入文件
        with open('./position_info.json', 'w') as f:
            json.dump(position_dict, f)

    def get_init_value(self):

        # 读取json文件中保存的最后一次的位姿信息
        with open('./position_info.json', 'r') as f:
            try:
                final_position = json.load(f)
                print final_position
            except ValueError:
                final_position = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        # self.x_d0 = round(final_position['x'], 2)
        # self.y_d0 = round(final_position['y'], 2)
        # self.theta_d0 = round(final_position['theta'], 2)
        self.x_d0 = final_position['x']
        self.y_d0 = final_position['y']
        self.theta_d0 = final_position['theta']
        # self.theta_0 = self.theta_d0

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

    def shutdown(self):
        # 使用cancel()停止定时器
        for i in threading.enumerate():
            if type(i) == threading._Timer:  # 如果在子线程中用到了Timer线程，必须先在子线程中杀死Timer线程，才能正常退出程序
                i.cancel()
        # Always stop the robot when shutting down the node
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())

        # 将最后一次结果写入json文件
        self.write_final_value()

        rospy.sleep(1)


if __name__ == '__main__':
    try:
        NavSquare()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation terminated.")

