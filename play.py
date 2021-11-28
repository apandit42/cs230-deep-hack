from pathlib import Path
import pickle
import time
import os
import nle
from nle.env.base import NLE
import gym
from simtest import TerminalStream

file = 'mon-hum-neu-mal 22:42:23 11-27-21.pickle'
block = pickle.loads((Path('nh-runs') / Path(file)).read_bytes())

frames = block['frame_counter']
tty_chars_stack = block['chars_stack']
tty_colors_stack = block['colors_stack']

X = NLE()

for chars, colors in zip(tty_chars_stack, tty_colors_stack):
    ansi_text = X.get_tty_rendering(chars, colors)
    
    break

# display.clear_output(wait=True)
#Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 22:10:14 11-27-21.pickle
#Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 21:55:22 11-27-21.pickle
# Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 22:38:18 11-27-21.pickle
#Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 22:42:23 11-27-21.pickle