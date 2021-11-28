from pathlib import Path
import pickle
import time
import os
from nle.nethack import tty_render
from ansitoimg.render import ansiToSVG

file = 'mon-hum-neu-mal 02:41:42 11-28-21.pickle'
block = pickle.loads((Path('nh-runs') / Path(file)).read_bytes())

frames = block['frame_counter']
tty_chars_stack = block['chars_stack']
tty_colors_stack = block['colors_stack']

for chars, colors in zip(tty_chars_stack, tty_colors_stack):
    ansi_text = tty_render(chars, colors)
    print(ansi_text)
    ansiToSVG(ansi_text, 'test.svg')
    break

# display.clear_output(wait=True)
#Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 22:10:14 11-27-21.pickle
#Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 21:55:22 11-27-21.pickle
# Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 22:38:18 11-27-21.pickle
#Documents/cs230-deep-hack/nh-runs/mon-hum-neu-mal 22:42:23 11-27-21.pickle