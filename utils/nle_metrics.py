from pathlib import Path
from .terminal_stream import TerminalStream
from .nethackboost import NetHackBoost
import nle
import pandas as pd
import matplotlib.pyplot as plt

class NetHackMetricsEnv():
    # must supply directory to store episodes
    def __init__(self, episodes_dir, character='valkyrie-dwarf', action_mode='reduced'):
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
        self.env = NetHackBoost(character, action_mode)
        self.action_space_size = self.env.action_space.n
        self.tty_stream = TerminalStream(self.env, save_dir=episodes_dir)
        self.episodes_dir = Path(episodes_dir)
        self.episodes_dict = {}
    
    # must supply a name for an episode
    def start(self, new_episode_name):
        self.episode_name = new_episode_name
        obs = self.env.reset()
        self.tty_stream.set_run(self.episode_name)
        self.tty_stream.record(self.env, 0)
        return obs
    
    # must provide action to step
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.tty_stream.record(self.env, reward)
        return obs, reward, done, info
    
    # render method
    def render(self):
        self.env.render()
    
    # finish, pass up the statistics
    def finish(self):
        summary, run_name = self.tty_stream.finish()
        self.episodes_dict[run_name] = summary
    
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