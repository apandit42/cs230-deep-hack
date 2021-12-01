from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import sys
import os
import nle
import gym

class TerminalStream():
    def __init__(self, env, save_dir='nh-runs'):
        # Indices for getting terminal screen data
        self.tty_chars_idx = env._observation_keys.index('tty_chars')
        self.tty_colors_idx = env._observation_keys.index('tty_colors')
        self.tty_cursor_idx = env._observation_keys.index('tty_cursor')
        self.blstats_idx = env._observation_keys.index('blstats')
        
        # Total frames collected
        self.frame_counter = 0
        
        # List of arrays holding collected data, empty rn
        self.tty_chars_stack = []
        self.tty_colors_stack = []
        self.tty_cursors_stack = []
        self.rewards = []

        # list to hold stats
        self.blstats = []
        # list of index labels
        self.blstats_labels = [
            'x',
            'y',
            'strength_percentage',
            'strength',
            'dexterity',
            'constitution',
            'intelligence',
            'wisdom',
            'charisma',
            'score',
            'hitpoints',
            'max_hitpoints',
            'depth',
            'gold',
            'energy',
            'max_energy',
            'armor_class',
            'monster_level',
            'experience_level',
            'experience_points',
            'time',
            'hunger_state',
            'carrying_capacity',
            'dungeon_number',
            'level_number',
        ]
        
        # If no savedir, then leave
        if save_dir is None:
            self.save_path = None
            return 
        
        # Otherwise, make sure path exists
        self.save_path = Path(save_dir)
        if not self.save_path.is_dir():
            self.save_path.mkdir(parents=True)
        
        # Finally, just set the correct function for ansi playback / recording
        try:
            from nle.nethack import tty_render as nle_tty_render
            self.tty_render_version = 'tty_render'
        except ImportError:
            print(f'Warning: tty_render function not found in nle.nethack. Searching in nle.env.base.')
            from nle.env.base import NLE
            self.tty_render_version = 'NLE'

    # reset character name, save path
    def set_run(self, run_name):
        self.run_name = run_name
        self.run_path = self.save_path / f'{self.run_name}.pickle'
    
    # drop data as long as save path set
    def save_data(self):
        # Exit if save path is n/a
        if self.save_path is None:
            return
        
        # Make a block with the data to save, and dump it as a pickle to the run folder
        save_block = {
            'frame_counter': self.frame_counter, 
            'chars_stack': self.tty_chars_stack, 
            'colors_stack': self.tty_colors_stack,
            'cursors_stack': self.tty_cursors_stack,
            'blstats': self.blstats,
            'rewards': self.rewards,
        }
        self.run_path.write_bytes(pickle.dumps(save_block))

        # Reset arrays to hold no data
        self.tty_chars_stack = []
        self.tty_colors_stack = []
        self.tty_cursors_stack = []
        self.rewards = []
        self.blstats = []
        self.frame_counter = 0

    def load_data(self, path):
        file_path = Path(path)
        save_block = pickle.loads(file_path.read_bytes())
        self.frame_counter = save_block['frame_counter']
        self.tty_chars_stack = save_block['chars_stack']
        self.tty_colors_stack = save_block['colors_stack']
        self.tty_cursors_stack = save_block['cursors_stack']
        self.rewards = save_block['rewards']
        self.blstats = save_block['blstats']

    def record(self, env, reward):
        self.tty_chars_stack += [env.last_observation[self.tty_chars_idx].copy()]
        self.tty_colors_stack += [env.last_observation[self.tty_colors_idx].copy()]
        self.tty_cursors_stack += [env.last_observation[self.tty_cursor_idx].copy()]
        self.rewards += [reward]
        self.blstats += [env.last_observation[self.blstats_idx].copy()]
        self.frame_counter += 1
    
    def finish(self):
        # dud run
        if self.frame_counter < 2:
            return
        # don't bother saving anything
        if self.save_path is None:
            return

        # get metrics
        stats_arr = np.row_stack(self.blstats)
        stats_arr = stats_arr[:,:-1]
        stats_arr = np.hstack((stats_arr, np.array(self.rewards).reshape(-1,1)))
        stats_labels = self.blstats_labels + ['rewards']
        stats_df = pd.DataFrame(stats_arr, columns=stats_labels)

        # write out to the stats folder for the run
        stats_dir = self.save_path / self.run_name
        if not stats_dir.is_dir():
            stats_dir.mkdir()
        stats_df.to_csv(stats_dir / 'stats.csv', index=False)

        # with a context manager
        with plt.style.context('seaborn-darkgrid'):
            # save figs in stats_dir / *.png, with this size
            size = (15,7)

            # First, score
            plt.figure(figsize=size)
            plt.scatter(stats_df['time'], stats_df['score'], marker='.', color='mediumseagreen')
            plt.ylabel('Score')
            plt.xlabel('Timestep')
            plt.title(f'{self.run_name} Score vs. Timestep')
            plt.savefig(stats_dir / 'score_vs_timestep.png')
            plt.close()

            # Then, hitpoints
            plt.figure(figsize=size)
            plt.scatter(stats_df['time'], stats_df['hitpoints'], marker='.', color='red')
            plt.ylabel('Hit Points')
            plt.xlabel('Timestep')
            plt.title(f'{self.run_name} Hit Points vs. Timestep')
            plt.savefig(stats_dir / 'hp_vs_timestep.png')
            plt.close()

            # Then, gold
            plt.figure(figsize=size)
            plt.scatter(stats_df['time'], stats_df['gold'], marker='.', color='gold')
            plt.ylabel('Gold')
            plt.xlabel('Timestep')
            plt.title(f'{self.run_name} Gold vs. Timestep')
            plt.savefig(stats_dir / 'gold_vs_timestep.png')
            plt.close()

            # Then, experience points
            plt.figure(figsize=size)
            plt.scatter(stats_df['time'], stats_df['experience_points'], marker='.', color='royalblue')
            plt.ylabel('Experience Points')
            plt.xlabel('Timestep')
            plt.title(f'{self.run_name} Experience Points vs. Timestep')
            plt.savefig(stats_dir / 'xp_vs_timestep.png')
            plt.close()

            # Then, dungeon number
            plt.figure(figsize=size)
            plt.scatter(stats_df['time'], stats_df['dungeon_number'], marker='.', color='slategray')
            plt.ylabel('Dungeon Number')
            plt.xlabel('Timestep')
            plt.title(f'{self.run_name} Dungeon Number vs. Timestep')
            plt.savefig(stats_dir / 'dungeon_num_vs_timestep.png')
            plt.close()

            # Then, rewards
            plt.figure(figsize=size)
            plt.scatter(stats_df['time'], stats_df['rewards'], marker='.', color='violet')
            plt.ylabel('Rewards')
            plt.xlabel('Timestep')
            plt.title(f'{self.run_name} Rewards vs. Timestep')
            plt.savefig(stats_dir / 'rewards_vs_timestep.png')
            plt.close()

        # Finally, store some summary statistics
        summary = {
            'total_reward': np.sum(stats_df['rewards']),
            'total_timesteps': stats_df['time'].iloc[-2],
            'total_score': stats_df['score'].iloc[-2],
            'max_gold': np.max(stats_df['gold']),
            'max_xp': stats_df['experience_points'].iloc[-2],
            'max_dungeon': np.max(stats_df['dungeon_number'])
        }

        # save and store data    
        self.save_data()

        # return summary statistics, run name
        return summary, self.run_name
        
    def get_frame_text(self, tty_chars, tty_colors, tty_cursor):
        if self.tty_render_version == 'tty_render':
            from nle.nethack import tty_render as nle_tty_render
            return nle_tty_render(tty_chars, tty_colors, tty_cursor)
        else:
            from nle.env.base import NLE
            return NLE().get_tty_rendering(tty_chars, tty_colors)
    
    def print_frame(self, tty_chars, tty_colors, tty_cursor):
        print(self.get_frame_text(tty_chars, tty_colors, tty_cursor))
    
    def render(self, run_path, out_path):
        os.system(f'python utils/multi_render.py {run_path} {out_path}')
    
    def run_render_monitor(self, monitor_dir, out_dir):
        monitor_dir_path = Path(monitor_dir)
        out_dir_path = Path(out_dir)
        if not monitor_dir_path.is_dir() or not out_dir_path.is_dir():
            print(f'Path {monitor_dir_path} or Path {out_dir_path} is not directory!')
            return
        print(f'Daemon Mode Running...')
        try:
            while True:
                run_files = {x.stem for x in monitor_dir_path.iterdir() if x.is_file()}
                out_files = {x.stem for x in out_dir_path.iterdir() if x.is_file()}
                if len(run_files - out_files):
                    print('Found files to render!')
                    files_to_run = run_files - out_files
                    for x in files_to_run:
                        self.render(f'{x}.pickle')
                time.sleep(1)
        except KeyboardInterrupt:
            print('\nExiting now!')
            return

if __name__ == '__main__':
    # python terminal_stream.py --monitor src_dir dst_dir
    if sys.argv[1] == '--monitor':
        env = gym.make('NetHackScore-v0')
        x = TerminalStream(env)
        x.run_render_monitor(sys.argv[2], sys.argv[3])