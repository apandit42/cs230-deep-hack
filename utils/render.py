from pathlib import Path
import pickle
from ansitoimg.render import ansiToRaster
import sys
import tempfile
import os

# try to select the right tty_render version
try:
    from nle.nethack import tty_render
    RENDER_CLIENT = 'tty_render'
except ImportError:
    from nle.env.base import NLE
    RENDER_CLIENT = 'NLE'
    tty_render = NLE()

# read in block file, assumes in nh-runs
def read_block_file(block_file):
    block_path = Path(block_file)
    block = pickle.loads(block_path.read_bytes())
    return block['frame_counter'], block['chars_stack'], block['colors_stack'], block['cursors_stack']

# get the ansi text
def get_ansi_text(chars, colors, cursor):
    if RENDER_CLIENT == 'tty_render':
        return tty_render(chars, colors, cursor)
    else:
        return tty_render.get_tty_rendering(chars, colors)

# render series of PNGs to video
def render(block_file, frame_counter, chars_stack, colors_stack, cursors_stack):
    with tempfile.TemporaryDirectory() as tempdir:
        # first render all the PNGs
        for i in range(frame_counter):
            ansi_text = get_ansi_text(chars_stack[i], colors_stack[i], cursors_stack[i])
            ansiToRaster(ansi_text, f'{tempdir}/{i}.png', theme='utils/dracula24.yaml')
            print(f'Running {tempdir}/{i}.png ...')
        # then ffmpeg them together, writing out to nh-vids
        export_file = block_file.replace('.pickle', '.mp4')
        os.system(f"ffmpeg -r 24 -i {tempdir}/%d.png -pix_fmt yuv420p nh-vids/{export_file}")

if __name__ == '__main__':
    # program expects second argument to be pickle file name to read
    print(f'RENDER: {RENDER_CLIENT}')
    block_file = sys.argv[1]
    frame_counter, chars_stack, colors_stack, cursors_stack = read_block_file(block_file)
    render(block_file, frame_counter, chars_stack, colors_stack, cursors_stack)