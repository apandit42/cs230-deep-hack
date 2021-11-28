from pathlib import Path
import pickle
import gym
import random
import nle
import numpy as np
from datetime import datetime

class RandomAgent():
    def __init__(self):
        self.actions = list(range(10))
    
    def act(self):
        return random.choice(self.actions)
    
class TerminalStream():
    def __init__(self, env, save_dir='nh-runs'):
        # Indices for getting terminal screen data
        self.tty_chars_idx = env._observation_keys.index('tty_chars')
        self.tty_colors_idx = env._observation_keys.index('tty_colors')
        self.tty_cursor_idx = env._observation_keys.index('tty_cursor')
        
        # Total frames collected
        self.frame_counter = 0
        
        # List of np arrays holding collected data, empty rn
        self.tty_chars_stack = []
        self.tty_colors_stack = []
        self.tty_cursor_stack = []
        
        # If no savedir, then leave
        if save_dir is None:
            self.save_path = None
            return 
        
        self.save_path = Path(save_dir)
        if not self.save_path.is_dir():
            self.save_path.mkdir(parents=True)
        
        self.run_name = f'{env.character} {datetime.now().strftime("%X %x").replace("/","-")}'
        self.run_path = self.save_path / Path(f'{self.run_name}.pickle')

    def save_data(self):
        # Exit if save path is n/a
        if self.save_path is None:
            return
        
        # Make a block with the data to save, and dump it as a pickle to the run folder
        save_block = {'chars_stack': self.tty_chars_stack, 'colors_stack': self.tty_colors_stack, 'cursor_stack': self.tty_cursor_stack}
        run_block = self.run_path
        run_block.write_bytes(pickle.dumps(save_block))

        # Reset arrays to hold only last data
        self.tty_chars_stack = []
        self.tty_colors_stack = []
        self.tty_cursor_stack = []
    
    def load_data(self, path):
        pass
    
    def record(self, env):
        self.tty_chars_stack += [env.last_observation[self.tty_chars_idx]]
        self.tty_colors_stack += [env.last_observation[self.tty_colors_idx]]
        self.tty_cursor_stack += [env.last_observation[self.tty_cursor_idx]]    
        self.frame_counter += 1
            
    def finish(self):
        if self.save_path is not None:
            self.save_data()
    
    def play(self):
        pass
    
    def export(self, export_dir):
        pass

class GameEnv():
    def __init__(self, agent):
        self.agent = agent
        self.env = gym.make('NetHackScore-v0')
        self.tty_stream = TerminalStream(self.env)
    
    def reset(self):
        self.env.reset()
        self.tty_stream.record(self.env)
    
    def step(self):
        obs, reward, done, info = self.env.step(self.agent.act())
        return obs, reward, done
    
    def run(self, total_steps = int(1e6)):
        self.reset()
        for i in range(total_steps):
            obs, reward, done = self.step()
            self.tty_stream.record(self.env)
            if done:
                self.reset()
        self.tty_stream.finish()

if __name__ == '__main__':
    agent = RandomAgent()
    game = GameEnv(agent)
    game.run()
    