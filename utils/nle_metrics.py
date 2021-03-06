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
from nle.nethack import Command


class NetHackMetricsEnv():
    # must supply directory to store episodes
    def __init__(self, episodes_dir, character='valkyrie-dwarf', actions_mode='reduced', reward_mode='base', test_mode=False, seed_csv='utils/seeds.csv'):
        """
        Note - Character codes from nethackboost.NetHackBoost
            # Character strings
        CHARACTER_CODES = {
            'valkyrie-dwarf': 'val-dwa-law-fem',
            'wizard-elf': 'wiz-elf-cha-mal',
            'cavewoman': 'cav-hum-neu-fem',
            'ranger-elf': 'ran-elf-cha-mal',
        }
        actions_mode = 'reduced' | 'with_menu' | 'base'
        reward_mode = 'base' | 'boost'
        """
        # Set the environment
        self.env = NetHackBoost(character=character, actions_mode=actions_mode, reward_mode=reward_mode)

        # setup action space to reflect the same setup as gym nethack
        ActionSpace = namedtuple('action_space', ['actions_mode', 'n', 'actions_list'])
        self.action_space = ActionSpace(actions_mode, self.env.action_space.n, self.env.actions_list)
        # Expose observation space for Custom-Agent Model
        self.observation_space = self.env.observation_space

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
            new_episode_name = datetime.now().strftime('%H-%M-%S-%f')
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
    
    # close, to preserve compatibility, will also write graphs
    def close(self):
        # just calls finish, writes last data, reports don't work
        # self.write_report()
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
    
    def expand_alpha_ranges(self, msg):
        while '-' in msg:
            range_start = msg.find('-') - 1
            range_end = range_start + 2
            msg = msg[:range_start] + string.ascii_letters[string.ascii_letters.index(msg[range_start]):string.ascii_letters.index(msg[range_end])] + msg[range_end:]
        return msg

    def get_message_menu(self, message_bytes):
        # Parse into str
        message_str = self.get_unicode_from_bytes(message_bytes)
        # Look for menu actions in message prompt, and capture them
        match_results = re.search(r'\s+\[(?:-)? ?([\$\-a-zA-Z]*) ?(?:or \?\*)?\]', message_str)
        if match_results:
            menu_actions = match_results.group(0)
            menu_actions = self.expand_alpha_ranges(menu_actions)
            return list(menu_actions)
        # look to see if menu actions prompting direction are in message prompt, and capture them
        elif re.search(r'In what direction\?'):
            menu_actions = 'kljhunby' # cardinal directions string
            return list(menu_actions)
        else:
            return []
    
    def get_screen_menu(self, tty_chars):
        # cut down on screen
        tty_str = self.get_unicode_from_bytes(tty_chars[1:22,20:])
        inv_key_regex = re.compile(r'([a-zA-Z]) - \w (?:\w+ ?)+')
        return inv_key_regex.findall(tty_str)
    
    # now mask the remaining items
    def mask_menu_items(self, mask, menu):
        # get the menu items
        menu_char_vals = [ord(x) for x in menu]
        raw_keys_index = [NetHackBoost.RAW_KEY_ACTIONS.index(x) for x in menu_char_vals]
        # Shift indices over
        menu_keys_index = np.array(raw_keys_index) + len(NetHackBoost.REDUCED_ACTIONS)
        # set all commands to 0
        mask[:] = 0
        # set menu keys back to 1
        mask[menu_keys_index] = 1
        # Now add in space + ESC
        mask[[NetHackBoost.REDUCED_ACTIONS.index(ord(' ')), NetHackBoost.REDUCED_ACTIONS.index(Command.ESC)]] = 1
        # return
        return mask

    # now, time to get mask
    def get_action_mask(self, obs):
        # start enabling everything
        mask = np.ones(self.action_space.n)
        # enable menu navigation interface with only the with_menu system
        if self.action_space.actions_mode == 'with_menu':
            message_menu = self.get_message_menu(obs['message'])
            # if both of these fail, no masking happens
            if message_menu:
                # disable and set commands
                mask = self.mask_menu_items(mask, message_menu)
                return mask
            screen_menu = self.get_screen_menu(obs['tty_chars'])
            if screen_menu:
                # disable and set commands
                mask = self.mask_menu_items(mask, screen_menu)
                return mask
        # return the final mask
        return mask   

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