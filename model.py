import keras
import numpy as np
import random
from memory import IndividualMemory

class atari_model:
    def __init__(self, n_actions):
        # We assume a theano backend here, so the "channels" are last.
        # 4 frames each 105x80 pixels and are greyscale so no need to worry about RGB
        ATARI_SHAPE = (105, 80, 4)

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
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = keras.layers.Dense(n_actions)(hidden)
        # Finally, we multiply the output by the mask!
        filtered_output = keras.layers.multiply([output, actions_input])

        self.model = keras.models.Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        self.model.compile(optimizer, loss='mse')
    
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

    def choose_best_action(self, state):
        return self.model.predict(state)

    def q_iteration(self, env, state, iteration, memory):
        # Choose epsilon based on the iteration
        epsilon = self.get_epsilon_for_iteration(iteration)

        # Choose the action 
        # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
        is_done = False
        while not is_done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = choose_best_action(self.model, state)

            new_frame, reward, is_done, _ = env.step(action)
            mem = IndividualMemory(state, action, new_frame, reward, is_done)
            memory.add(mem)
            env.render()

        # Sample and fit
        batch = memory.sample_batch(32)
        start_states = np.array([i.start_state for i in batch])
        actions = np.array([i.action for i in batch])
        actions_shape = (actions.size, actions.max()+1)
        actions = np.zeros(actions_shape)
        rewards = np.array([i.reward for i in batch])
        next_states = np.array([i.next_state for i in batch])
        is_terminal = np.array([i.is_done for i in batch])

        self.fit_batch(0.99, start_states, actions, rewards, next_states, is_terminal) 

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
        next_Q_values = model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.model.fit(
            [start_states, actions], actions * Q_values[:, None],
            nb_epoch=1, batch_size=len(start_states), verbose=0
        )