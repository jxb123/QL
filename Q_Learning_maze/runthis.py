

from Q_Learning_maze.maze_env import Maze
from Q_Learning_maze.RL_brain import QLearningTable

def update():
    #运行100个回合
    for episode in range(100):
        #环境的观测值
        observation = env.reset()
        while True:
            #刷新环境
            env.render()
            #基于观测值来挑选动作
            action = RL.choose_action(str(observation))
            #根据观测值来得到下一个观测值和reword，done是看是否跳进地狱
            observation_,reward,done = env.step(action)
            #RL从该transition中学习
            RL.learn(str(observation),action,reward,str(observation_))
            observation = observation_
            if done:
                break
    #结束游戏
    print('game over')
    env.destroy()
if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()