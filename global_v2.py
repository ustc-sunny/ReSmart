# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:30:06 2023

@author: 94481
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import beta
#GS UCB TS

class Arm:
    def __init__(self, arm_id, num_players):
        self.arm_id = arm_id
#         self.matched_players = []
        self.player_preference = np.random.permutation(np.arange(0, num_players)) #player_preference第i个表示第i个player的排名，越小越好


class Market:
    def __init__(self, num_players, num_arms, num_time_slots, r):
        self.num_players = num_players
        self.num_arms = num_arms
        self.num_time_slots = num_time_slots
        self.arms = [Arm(i, self.num_players) for i in range(self.num_arms)]
        self.choosing_result = np.full((self.num_time_slots, self.num_players), -1)  # 用于记录每个时间步的选择结果
        self.choosing_result[0] = np.random.randint(0, self.num_arms, self.num_players)
        self.matching_result = np.full((self.num_time_slots, self.num_players), -1)  # 初始化matching_result矩阵
        self.matching_result[0] = self.choosing_result[0]
        self.alpha = np.ones((self.num_players, self.num_arms))
        self.beta = np.ones((self.num_players, self.num_arms))
        self.theta = np.zeros((self.num_players, self.num_arms))
        self.k = 1
        self.L = 100
        self.feasible_set = [[] for _ in range(num_players)]
        self.r= r
        self.X = np.zeros((num_players, num_arms))
        self.Y = np.zeros((num_players, num_arms))

        self.regrets = [[] for _ in range(num_players)]

    def update_feasible_set(self, time_slot):
        self.feasible_set = [[] for _ in range(self.num_players)]
        #将当前匹配的每个arm的最大preference所对应的player_id输出成一维矩阵
        current_max_arm_pre = [-1] * self.num_arms # 初始化为-1
        for i in range(self.num_players):
            mi = self.matching_result[time_slot -1][i]#############################time_slot == 1 的时候
            if mi == -1:#if no match for player i
                continue
            max_mj = current_max_arm_pre[mi]
            if max_mj == -1:
                current_max_arm_pre[mi] = i
            elif np.where(self.arms[mi].player_preference == i)[0] < np.where(self.arms[mi].player_preference == max_mj)[0]:
                current_max_arm_pre[mi] = i
        #start to update feasible set
        for j in range(self.num_arms):
            if current_max_arm_pre[j] != -1:
                k = 0
                while self.arms[j].player_preference[k] != current_max_arm_pre[j]:
                    #print(self.arms[j].player_preference[k])
                    self.feasible_set[self.arms[j].player_preference[k]].append(j)
                    k += 1

    def run_market(self):
        for time_slot in range(1, self.num_time_slots):
            print('&&&')
            #initialization
            #Update the feasible set
            self.update_feasible_set(time_slot)
            print(self.feasible_set)
            self.X = np.zeros((self.num_players, self.num_arms))
            self.Y = np.zeros((self.num_players, self.num_arms))
            #start algorithm
            if self.k == 1:
               #Set 𝑛𝑡𝑖,𝑗 = 0, 𝛼𝑖,𝑗 = 1, 𝛽𝑖,𝑗 = 1, ∀𝑖 ∈ T, ∀𝑗 ∈ U
               #####nij??
               self.alpha = np.ones((self.num_players, self.num_arms))
               self.beta = np.ones((self.num_players, self.num_arms))
               #∀𝑖 ∈ T, 𝑎1 (𝑖) = 𝑗, 𝑗 ∼ U uniformly at random.
               self.choosing_result[0] = np.random.randint(0, self.num_arms, self.num_players)
               self.matching_result[0] = self.choosing_result[0]
            #Distributed Task Matching using ThompsonSampling (DTTS) start.
            for i in range(self.num_players):
                #𝜃𝑖,𝑗 ∼ 𝐵𝑒 (𝛼𝑖,𝑗, 𝛽𝑖,𝑗)
                for j in range(self.num_arms):
                    self.theta[i][j] = beta(self.alpha[i][j], self.beta[i][j]).rvs()
                #Draw 𝐵𝑖(𝑡) ∼ 𝐵𝑒𝑟(𝜆) independently.
                lambda_prob = 0.2    #lambda
                if random.random() < lambda_prob:#lambda
                    #set at(i) = at-1(i)
                    self.choosing_result[time_slot][i] = self.choosing_result[time_slot - 1][i]
                else:
                    #updating_feasible_set is in initialization
                    #Attempt to propose the matching request to 𝑎 ∈ 𝐹𝑡 (𝑖) with the maximum 𝜃𝑖,𝑗 .
                    if not self.feasible_set:
                        self.choosing_result[time_slot][i] = self.choosing_result[time_slot - 1][i]
                    else:
                        a = 0
                        for j in range(self.num_arms):
                            if j in self.feasible_set[i] and self.theta[i][j] > self.theta[i][a]:
                                a = j
                        #Set 𝑎𝑡 (𝑖) = 𝑎
                        self.choosing_result[time_slot][i] = a
                #if 𝜋¯𝑎𝑡 (𝑖)(𝑖) ≻𝑎𝑡 (𝑖) 𝜋¯𝑖′ (𝑖),𝑖′ ∈ T𝑡𝑖,j
                if self.arms[self.choosing_result[time_slot][i]].player_preference[0] == i:
                    self.matching_result[time_slot][i] = a
                    #Obtain a utility 𝑋t(i, mt(i))，正态分布
                    #self.X[i][a] = random.gauss(self.r[i][a], 1)########正太分布大于1不能后续用伯努利
                    #draw 𝑌𝑡(i, mt(i))
                    #print(self.X[i][a])
                    if random.random() < self.r[i][a]:
                        self.Y[i][a] = 1
                    else:
                        self.Y[i][a] = 0
                    #Update parameter of Beta distribution 𝐵𝑒:
                    self.alpha[i][a] += self.Y[i][a]
                    self.beta[i][a] = self.beta[i][a] + 1 - self.Y[i][a]
                #update regret
                #if not self.regrets[i]:
                #    self.regrets[i].append(self.r[i][])
            if self.k <= self.L:########self.L???
                self.k += 1
            else:
                self.k = 1

def main():
    # 设置参数
    num_players = 5
    num_arms = 5
    num_episodes = 1
    sigle_time_slots = 10
    stable_index = []
    num_time_slots = sigle_time_slots * num_episodes
    # for round in range(num_instance):    
    #     # 创建市场并运行
    #     print(round)
    #     market = Market(num_players, num_arms, num_episodes, num_time_slots, gamma)
    #     market.run_market()
    #     stable_index.append(market.stable_index)
        
    
    # stable_sum = np.zeros(num_episodes)
    # for round in range(num_instance):
    #     instance = stable_index[round]
    #     stable_sum = stable_sum + instance
        
    # stable_mean = stable_sum/num_instance
    
    # time_slots = range(1, num_episodes + 1)  # 时间槽
    # plt.plot(time_slots, stable_mean, label='N=5')
    # plt.xlabel('Episode')
    # plt.ylabel('Stable probability')
    # plt.title('Stable probability')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    r = np.random.uniform(low=0, high=1, size=(5, 5))
    market = Market(num_players, num_arms, num_time_slots, r)
    market.run_market()
    #stable_index.append(market.stable_index)
    # 平均累积奖励柱状图
    # 准备数据
    # x_data = range(1, num_episodes * num_time_slots)
    # y_TS_UCB = market.Average_cumulative_reward
    # y_TS = market.Average_cumulative_reward
    # y_UCB = market.Average_cumulative_reward
    # #1.将x轴转换为数值
    # x = np.arange(len(x_data))
    # #2.设置图形的宽度
    # width = 0.2
    # #_______________确定x起始位置
    # #TS-UCB起始位置
    # x_TS_UCB = x - width
    # #TS起始位置
    # x_TS = x
    # #UCB起始位置
    # x_UCB = x + width
    # #_______________分别绘制图形
    # #TS-UCB图形
    # plt.bar(x_TS_UCB, y_TS_UCB, width = width, label = 'TS-UCB', hatch = "...", color = 'w', edgecolor = "k")
    # #TS图形
    # plt.bar(x_TS, y_TS, width = width, label = 'TS', hatch = "++", color = 'w', edgecolor = "k")
    # #UCB图形
    # plt.bar(x_UCB, y_UCB, width = width, label = 'UCB', hatch = "XX", color = 'w', edgecolor = "k")
    # plt.title("Average cumulative reward")
    # plt.xlabel('Timeslot')
    # plt.ylabel('Average cumulative reward')
    # # 显示
    # plt.show()

    # 平均stable regret折线图
    # 平均stable regret折线图
    # x_data = range(1, num_time_slots)
    # y_data = market.Regret
    # plt.plot(x_data, y_data, linewidth=1, label="1")
    # plt.title("Average stable regret")
    # plt.xlabel('timeslot')
    # plt.ylabel('Average stable regret')
    # plt.show
main()