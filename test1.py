import numpy as np
import pandas as pd
import time
#产生伪随机序列，即随机效果是一样的
np.random.seed(2)
#全局变量
N_STATES = 6 #开始距离离宝藏的距离有6步
ACTIONS = ['left','right'] #可以选择的动作
EPSILON = 0.9 #greedy police
ALPHA = 0.1 #学习率
LAMBDA = 0.9 #未来奖励的衰减值
MAX_EPISODES = 13 #只玩13回合，13回合后训练的很好了
FRESH_TIME = 0.03 #走一步所话的时间

def build_q_table(n_states,actions):
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))), #全0初始化
        columns=actions,
    )
    #print(table)
    return table

def choose_action(state,q_table):
    state_actions = q_table.iloc[state,:] #选状态下的所有值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name
#环境的反馈
def get_env_feedback(S,A):
    if A == 'right':
        if S == N_STATES -2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_,R
def update_env(S,episode,step_counter):
    env_list = ['-'] * (N_STATES-1) + ['T'] # '------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                   ',end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def rl():
    #创建一个q——table
    q_table = build_q_table(N_STATES,ACTIONS)
    #从第一个回合玩到最后一个回合
    for episode in range(MAX_EPISODES):
        step_counter = 0
        #初始状态将探索者放在最左边
        S = 0
        is_terminated = False
        #第一步更新环境
        update_env(S,episode,step_counter)
        while not is_terminated:
            #先给一个action
            A = choose_action(S,q_table)
            #根据初始action和state得到下一个action和reword
            S_,R = get_env_feedback(S,A)
            #估计值
            q_predict = q_table.loc[S,A]
            if S_ != 'terminal':
                #真实值
                q_target = R + LAMBDA * q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S,A] += ALPHA * (q_target - q_predict)
            S = S_
            #后面每步再更新环境
            update_env(S,episode,step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
