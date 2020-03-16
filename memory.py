from random import randint
import numpy as np
import h5py

class IndividualMemory:
    def __init__(self, start_state, action, new_state, reward, is_done):
        self.start_state = start_state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.is_done = is_done

class RingBuf:
    def __init__(self, size):
        # allocate one extra element, this way, self.start == self.end always means 
        # the buffer is EMPTY, whereas if you allocate exactly the right number of 
        # elements, it could also mean the buffer is full. 
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        self.n_valid_elements = 0
        self.size = size

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)

        if self.n_valid_elements < self.size:
            self.n_valid_elements += 1

        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def sample_batch(self, num_of_elements):
        batch = []
        for i in range(num_of_elements):
            batch.append(self.__getitem__(randint(0, self.n_valid_elements-1)))
        return batch

    def save_memory(self):
        save_file_path = "buffer_memory/saved_memories.hdf5"
        f = h5py.File(save_file_path, "w")

        start_state = []
        action = []
        new_state = []
        reward = []
        is_done = []
        for i in range(self.n_valid_elements):
            mem = self.__getitem__(i)

            start_state.append(mem.start_state)
            action.append(mem.action)
            new_state.append(mem.new_state)
            reward.append(mem.reward)
            is_done.append(mem.is_done)


        data_set = f.create_dataset("start_state", data=start_state)
        f.create_dataset("action", data=action)
        f.create_dataset("new_state", data=new_state)
        f.create_dataset("reward", data=reward)
        f.create_dataset("is_done", data=is_done)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]