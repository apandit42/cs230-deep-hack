from pathlib import Path
from datetime import datetime
from .terminal_stream import TerminalStream
from .nethackboost import NetHackBoost
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
import random
import sys
import numpy as np
import re
import string


class NetHackMetricsEnv():
    # must supply directory to store episodes
    def __init__(self, episodes_dir, character='valkyrie-dwarf', actions_mode='reduced', test_mode=False, seed_csv='utils/seeds.csv'):
        """
        Note - Character codes from nethackboost.NetHackBoost
            # Character strings
        CHARACTER_CODES = {
            'valkyrie-dwarf': 'val-dwa-law-fem',
            'wizard-elf': 'wiz-elf-cha-mal',
            'cavewoman': 'cav-hum-neu-fem',
            'ranger-elf': 'ran-elf-cha-mal',
        }
        """
        # Set the environment
        self.env = NetHackBoost(character=character, actions_mode=actions_mode)

        # setup action space to reflect the same setup as gym nethack
        ActionSpace = namedtuple('action_space', ['actions_mode', 'n', 'action_list'])
        self.action_space = ActionSpace(actions_mode, self.env.action_space.n, self.env.REDUCED_ACTIONS if actions_mode == 'reduced' else self.env.REDUCED_ACTIONS_WITH_MENU)

        # set the test mode and init seeds, init state
        self.init_seeds(seed_csv)
        self.test_mode = test_mode
        # init won't be complete until first reset or start call
        self.init_complete = False

        # Set the recorder
        self.tty_stream = TerminalStream(self.env, save_dir=episodes_dir)
        self.episodes_dir = Path(episodes_dir)
        self.episodes_dict = {}
    
    # init seeds
    def init_seeds(self, seed_csv):
        if Path(seed_csv).is_file():
            seed_csv_df = pd.read_csv(seed_csv)
        else:
            # NOTE: Total of 20K seeds, 2 seeds used per run
            TOTAL_SEEDS = 20000
            TEST_SEEDS = 1000
            rng = random.SystemRandom()
            generated_seeds = [rng.randrange(sys.maxsize) for x in range(TOTAL_SEEDS)]
            test_seed_idxs = set(random.sample(list(range(TOTAL_SEEDS)), TEST_SEEDS))
            seed_csv_df = pd.DataFrame({'seed': generated_seeds, 'is_test_seed': [True if i in test_seed_idxs else False for i in range(TOTAL_SEEDS)]})
            seed_csv_df.to_csv(seed_csv, index=False)
        # now, just set df
        self.seed_csv = seed_csv_df
    
    # reset, to preserve compatibility with 
    def reset(self, new_episode_name=None):
        # if tty_stream actually has data, call finish, otherwise just reset
        if self.tty_stream.stream_has_data():
            self.finish()
        # if no new episode name is passed in, use time info
        if new_episode_name is None:
            new_episode_name = datetime.now().strftime('%f%M%S')
        return self.start(new_episode_name)

    # must supply a name for an episode
    def start(self, new_episode_name):
        # change episode name
        self.episode_name = new_episode_name

        # sample from seeds, depending on truth value of test mode
        core_seed, disp_seed = self.seed_csv[self.seed_csv.is_test_seed == self.test_mode].sample(2).seed
        # set underlying environment
        self.env.seed(core=core_seed, disp=disp_seed)
        # now save those seeds to instance for a bit till finish called
        self.core_seed = core_seed
        self.disp_seed = disp_seed
        
        # now, new run should be using new seeds
        obs = self.env.reset()
        self.tty_stream.set_run(self.episode_name)
        self.tty_stream.record(self.env, 0)
        return obs
        
    # must provide action to step
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # new function, for navigating menus
        obs['mask'] = self.get_action_mask(obs)
        self.tty_stream.record(self.env, reward)
        return obs, reward, done, info
    
    # finish, pass up the statistics
    def finish(self):
        summary, run_name = self.tty_stream.finish()
        self.episodes_dict[run_name] = summary
        # make sure to save seeds
        self.episodes_dict[run_name]['core_seed'] = self.core_seed
        self.episodes_dict[run_name]['disp_seed'] = self.disp_seed
    
    # close, to preserve compatibility
    def close(self):
        # just calls finish, writes last data
        self.finish()

    # render method
    def render(self):
        self.env.render()
    
    def get_unicode_from_bytes(self, bytes_array):
        unicode_str = ''
        if len(bytes_array.shape) == 2:
            for i in range(bytes_array.shape[0]):
                unicode_str += bytes(bytes_array[i,:]).decode('utf-8')
                unicode_str += '\n'
        else:
            unicode_str += bytes(bytes_array).decode('utf-8')
            unicode_str += '\n'
        return unicode_str
    
    def get_message_menu(self, message_bytes):
        # Parse into str
        message_str = self.get_unicode_from_bytes(message_bytes)
        # Look for menu actions in message prompt, and capture them
        match_results = re.search(r'\s+\[(?:-)? ?([\$\-a-zA-Z]*) ?(?:or \?\*)?\]', message_str)
        if match_results:
            menu_actions = match_results.group(0)
        
        # otherwise, look for directions prompt
        if re.search(r'In what direction\?'):
            menu_actions = 'kljhunby' # cardinal directions string
    
    # now, time to get mask
    def get_action_mask(self, obs):
        # define where player stats begin
        stats_row = 22
        # simple case of no menu navigation
        if self.action_space.actions_mode == 'reduced':
            return np.ones(self.action_space.n)

    # write out the pandas summary, and graph
    def write_report(self):
        episodes_df = pd.DataFrame(self.episodes_dict).T
        episodes_df.to_csv(self.episodes_dir / 'summary.csv')

        # cut to max graphable
        if episodes_df.shape[0] > 35:
            episodes_df = episodes_df.iloc[-35:,:]
        
        with plt.style.context('seaborn-darkgrid'):
            # size of the graph
            size = (15,7)

            # Now the graphs
            # Total Rewards
            plt.figure(figsize=size)
            plt.xticks(rotation=90)
            plt.bar(episodes_df.index, episodes_df['total_reward'])
            plt.ylabel('Total Reward')
            plt.xlabel('Episode',labelpad=20)
            plt.title('Total Reward vs. Episode')
            plt.savefig(self.episodes_dir / 'total_rewards.png')

            # Total Timesteps
            plt.figure(figsize=size)
            plt.xticks(rotation=90)
            plt.bar(episodes_df.index, episodes_df['total_timesteps'])
            plt.ylabel('Total Timesteps')
            plt.xlabel('Episode',labelpad=20)
            plt.title('Total Timesteps vs. Episode')
            plt.savefig(self.episodes_dir / 'total_timesteps.png')

            # Total Score
            plt.figure(figsize=size)
            plt.xticks(rotation=90)
            plt.bar(episodes_df.index, episodes_df['total_score'])
            plt.ylabel('Total Score')
            plt.xlabel('Episode',labelpad=20)
            plt.title('Total Score vs. Episode')
            plt.savefig(self.episodes_dir / 'total_score.png')

            # Max Gold
            plt.figure(figsize=size)
            plt.xticks(rotation=90)
            plt.bar(episodes_df.index, episodes_df['max_gold'])
            plt.ylabel('Max Gold')
            plt.xlabel('Episode',labelpad=20)
            plt.title('Max Gold vs. Episode')
            plt.savefig(self.episodes_dir / 'max_gold.png')

            # Max XP
            plt.figure(figsize=size)
            plt.xticks(rotation=90)
            plt.bar(episodes_df.index, episodes_df['max_xp'])
            plt.ylabel('Max Experience')
            plt.xlabel('Episode',labelpad=20)
            plt.title('Max Experience vs. Episode')
            plt.savefig(self.episodes_dir / 'max_experience.png')

            # Max Dungeon
            plt.figure(figsize=size)
            plt.xticks(rotation=90)
            plt.scatter(episodes_df.index, episodes_df['max_dungeon'])
            plt.ylabel('Max Dungeon')
            plt.xlabel('Episode',labelpad=20)
            plt.title('Max Dungeon vs. Episode')
            plt.savefig(self.episodes_dir / 'max_dungeon.png')