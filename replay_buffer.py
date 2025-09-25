from collections import deque
import random


class ReplayBuffer:
     """
    A simple FIFO buffer to store self-play game data.
    """
     
     def __init__(self,max_size:int):    
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

     def add(self, experience:tuple):
        # experience is a tuple (observation, policy, outcome)
        self.buffer.append(experience)

     def add_game_data(self, game_data:list):
        for experience in game_data:
            self.add(experience)

     def sample(self, batch_size:int) -> list:
        return random.sample(self.buffer, batch_size)    


     def __len__(self):
          """Returns the current size of the buffer."""
          return len(self.buffer)