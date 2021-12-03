import pandas as pd
from pathlib import Path
import sys
from multiprocessing import Pool
import matplotlib.pyplot as plt

def summarize_episode(episode_dir):
    episode_df = pd.read_csv(episode_dir / 'stats.csv')
    return (
        episode_dir.stem,
        {
            'total_reward': episode_df['rewards'].iloc[:-1].sum(),
            'total_timesteps': episode_df['time'].iloc[:-1].max(), 
            'max_depth': episode_df['depth'].iloc[:-1].max(),
            'max_score': episode_df['score'].iloc[:-1].max(),
            'max_gold': episode_df['gold'].iloc[:-1].max(),
            'max_xp': episode_df['experience_points'].iloc[:-1].max(),
        }
    )

def main(folder, source):
    source = Path(source)
    folder = Path(folder)
    episode_dirs = [x for x in folder.iterdir() if x.is_dir()]
    with Pool(12) as p:
        episode_summary_list = p.map(summarize_episode, episode_dirs)
    episode_dict = {episode_name: episode_summary for episode_name, episode_summary in episode_summary_list}
    episodes_df = pd.DataFrame(episode_dict).T
    try:
        episodes_df.index = pd.to_datetime(episodes_df.index, format='%H-%M-%S-%f')
    except ValueError:
        episodes_df = episodes_df.sort_index()
        episodes_df = episodes_df.reset_index()
    episodes_df.to_csv(f'{folder.stem}_SUMMARY.csv', index=False)

    figsize = (15,7)
    # plots of reward, score
    with plt.style.context('seaborn-darkgrid'):
        plt.figure(figsize=figsize)
        plt.scatter(episodes_df.index, episodes_df['total_timesteps'], marker='o', color='slategray')
        plt.title(f'Total Timesteps Per Episodes')
        plt.ylabel(f'Maximum Timesteps')
        plt.xlabel(f'Episode Number')
        plt.savefig(source / f'{folder.stem}_TIMESTEPS.png')
        
        plt.figure(figsize=figsize)
        plt.scatter(episodes_df.index, episodes_df['max_score'], marker='o', color='mediumseagreen')
        plt.title(f'Max Score Per Episode')
        plt.ylabel(f'Max Score')
        plt.xlabel(f'Episode Number')
        plt.savefig(source / f'{folder.stem}_SCORE.png')


if __name__ == '__main__':
    if '-s' in sys.argv:
        main(sys.argv[-1])
    elif '-a' in sys.argv:
        for x in Path(sys.argv[-1]).iterdir():
            if x.is_dir():
                main(x, Path(sys.argv[-1]))