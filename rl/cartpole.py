from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import argparse
import logging
import sys
import gym
from gym import wrappers


class DQNAgent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.model = self._build_model()
        self.memory = deque()
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_space.shape[0], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            act = self.action_space.sample()
            # print("Action Sample: ", act)
            return act
        act = self.model.predict(state)
        act = np.argmax(act[0])
        # print("Action Neural: ", act)
        return act

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                q_value = np.amax(self.model.predict(next_state)[0])
                target = reward + self.gamma * q_value
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    # args = parser.parse_args()

    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    # gym.undo_logger_setup()
    # logger = logging.getLogger()
    # formatter = logging.Formatter('[%(asctime)s] %(message)s')
    # handler = logging.StreamHandler(sys.stderr)
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    # logger.setLevel(logging.INFO)

    env = gym.make('CartPole-v1')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = '/tmp/dqn-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env.seed(0)
    agent = DQNAgent(env.observation_space, env.action_space)

    episode_count = 5000
    state_size = env.observation_space.shape[0]
    batch_size = 32
    max_t = 0

    for i in range(episode_count):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        t = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            t += 1
            if done:
                max_t = t if t > max_t else max_t
                print("episode: {}/{}, score: {}, e: {:.2}"
                       .format(i, episode_count, t, agent.epsilon))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        if len(agent.memory) >= batch_size:
            agent.replay(batch_size)

    # Close the env and write monitor result info to disk
    env.close() 
    print("Max score: ", max_t)
