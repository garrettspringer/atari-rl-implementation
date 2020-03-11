import keras
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from memory import IndividualMemory
from utils import preprocess, to_grayscale, transform_reward

class atari_model:
    def __init__(self, actions, n_actions):
        self.state_list = []
        self.actions = actions
        self.n_actions = n_actions
        self.episode_reward = 0
        self.reward_hiscore = 0
        self.load_file_path = "weights/"

        ATARI_SHAPE = (4, 105, 80)

        # With the functional API we need to define the inputs.
        frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
        actions_input = keras.layers.Input((n_actions,), name='mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = keras.layers.Lambda(lambda x: x / 255.0, output_shape=ATARI_SHAPE)(frames_input)
    
        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        conv_1 = keras.layers.convolutional.Conv2D(
            16, (8, 8), activation="relu", strides=(4, 4)
        )(normalized)
        # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
        conv_2 = keras.layers.convolutional.Conv2D(
            32, (4, 4), activation="relu", strides=(2, 2)
        )(conv_1)
        # Flattening the second convolutional layer.
        conv_flattened = keras.layers.core.Flatten("channels_first")(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(n_actions)(hidden)
        # Finally, we multiply the output by the mask!
        filtered_output = keras.layers.multiply([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer, loss='mse')

        #self._load_weights()

    def reset_episode_scores(self):
        if self.reward_hiscore < self.episode_reward:
            self.reward_hiscore = self.episode_reward
            print("New hiscore {}".format(self.reward_hiscore))
        self.episode_reward = 0
    
    def get_epsilon_for_iteration(self, iteration):
        """
        Returns the learning rate (epislon) for the iteration
        Scales from 1 to 0.1 linearly, then remains fixed at 0.1 
        """
        if (iteration >= 1000000):
            return 0.1
        elif (iteration == 0):
            return 1.0
        else:
            return (1111111.11-iteration)/1111111.11

    def previous_state(self, steps_back):
        if steps_back == -1:
            return self.state_list[len(self.state_list)-4:len(self.state_list)]
        elif steps_back == -2:
            return self.state_list[len(self.state_list)-8:len(self.state_list)-4]
        else:
            raise Exception("steps_back must be integers either -1 or -2")

    def choose_best_action(self, state):
        actions = np.ones((1, self.n_actions))

        output = self.model.predict([state, actions])
        return np.argmax(output)

    def run_game(self, env, ring_buf):
        frame = preprocess(env.reset())
        is_done = False
        self.state_list = [frame]

        # Need 4 Frames for initial state and following 4 Frames for next state
        for i in range(6):
            new_frame, reward, is_done, _ = env.step(env.action_space.sample())
            self.episode_reward += transform_reward(reward)
            self.state_list.append(preprocess(new_frame))
            env.render()

            if is_done:
                self.reset_episode_scores()
                break
            
        while not is_done:
            action = env.action_space.sample()
            new_frame, reward, is_done, _ = env.step(action)
            self.episode_reward += transform_reward(reward)
            self.state_list.append(preprocess(new_frame))
            env.render()

            mem = IndividualMemory(
                self.state_list[len(self.state_list)-8:len(self.state_list)-4], 
                action, 
                self.state_list[len(self.state_list)-4:len(self.state_list)], 
                transform_reward(reward), 
                is_done
            )
            ring_buf.append(mem)

            if is_done:
                self.reset_episode_scores()

    def warmup(self, env, num_games, my_ring_buf):
        print("Beginning Warmup")
        for i in range(num_games):
            self.run_game(env, my_ring_buf)
        print("Warmup Completed")

    def q_iteration(self, env, iteration, memory):
        # Get starting state
        is_done = False
        for i in range(3):
            action = env.action_space.sample()
            new_frame, reward, is_done, _ = env.step(action)
            self.episode_reward += transform_reward(reward)
            self.state_list.append(preprocess(new_frame))
            iteration+=1
            env.render()

        while not is_done:
            if random.random() < self.get_epsilon_for_iteration(iteration):
                action = env.action_space.sample()
            else:
                action = self.choose_best_action([self.previous_state(-1)])
            
            new_frame, reward, is_done, _ = env.step(action)
            self.episode_reward += transform_reward(reward)
            self.state_list.append(preprocess(new_frame))
            iteration+=1
            env.render()

            # Add memory if there is enough data to do so
            if len(self.state_list) > 7:
                mem = IndividualMemory(
                    self.previous_state(-2), 
                    action, 
                    self.previous_state(-1),
                    transform_reward(reward), 
                    is_done
                )
                memory.append(mem)

            # Sample and fit for experience replay at the end of each step
            sample_mem_size = 32
            if memory.n_valid_elements >= sample_mem_size:
                batch = memory.sample_batch(sample_mem_size)
                start_states = np.array([i.start_state for i in batch])

                actions = np.array([i.action for i in batch])
                actions_shape = np.zeros((actions.size, actions.max()+1))
                actions_shape[np.arange(actions.size), actions] = 1
                actions = actions_shape

                rewards = np.array([i.reward for i in batch])
                next_states = np.array([i.new_state for i in batch])
                is_terminal = np.array([i.is_done for i in batch])

                # FIXME make things like gamma into constants to be read in using JSON
                self.fit_batch(0.99, start_states, actions, rewards, next_states, is_terminal)
        
        self.reset_episode_scores()
        return iteration

    def fit_batch(self, gamma, start_states, actions, rewards, next_states, is_terminal):
        """Do one deep Q learning iteration.
        Params:
        - model: The DQN
        - gamma: Discount factor (should be 0.99)
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal
        """
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        print("fitting batch")
        next_Q_values = self.model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.model.fit(
            [start_states, actions], actions * Q_values[:, None],
            epochs=1, batch_size=len(start_states), verbose=0
        )

    def _load_weights(self):
        if not self.load_file_path:
            return
        path = self.load_file_path + "model.h5"
        self.model.load_weights(path)

    def _save_weights(self):
        if not self.load_file_path:
            return
        path = self.load_file_path + "model.h5"
        self.model.save_weights(path)