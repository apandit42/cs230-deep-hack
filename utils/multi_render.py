from pathlib import Path
import pickle
from multiprocessing import Pool
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

# read in block file, must be path from calling dir
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

# multi render
def render_worker(payload):
    file, chars, colors, cursors = payload
    ansi_text = get_ansi_text(chars, colors, cursors)
    ansiToRaster(ansi_text, file, theme='utils/dracula24.yaml')
    print(f'Rendering {file} ...')
    return file

# render series of PNGs to video
def render(out_file, frame_counter, chars_stack, colors_stack, cursors_stack, start_frame=0, stop_frame=None, step=1):
    print(f'Settings applied:')
    print(f'outfile: {out_file}, frame_counter: {frame_counter}, start_frame: {start_frame}, stop_frame: {stop_frame}, and step: {step} ...')
    if stop_frame is None:
        stop_frame = frame_counter
    start_frame = int(start_frame)
    stop_frame = int(stop_frame)
    step = int(step)
    with tempfile.TemporaryDirectory() as tempdir:
        payload = [(f'{tempdir}/{i}.png', chars_stack[j], colors_stack[j], cursors_stack[j]) for i,j in enumerate(list(range(start_frame, stop_frame, step)))]
        with Pool(24) as p:
            file_list = p.map(render_worker, payload)
        print(f'Rendered {len(file_list)} PNGs...')
        
        os.system(f"/usr/bin/ffmpeg -r 24 -i {tempdir}/%d.png -vcodec libx264 -crf 15 -pix_fmt yuv420p {out_file}")
        print(f'Completed export of {out_file}!')

# other function that readies main
def main(block_file, out_file, optional=None):
    print(f'RENDER: {RENDER_CLIENT}')
    frame_counter, chars_stack, colors_stack, cursors_stack = read_block_file(block_file)
    if optional:
        render(out_file, frame_counter, chars_stack, colors_stack, cursors_stack, start_frame=optional.get('start_frame',0), stop_frame=optional.get('stop_frame',None), step=optional.get('step',1))
    else:
        render(out_file, frame_counter, chars_stack, colors_stack, cursors_stack)

if __name__ == '__main__':
    # program expects second argument to be pickle file name to read, and then second file name to be 
    if '-h' in sys.argv or '--help' in sys.argv:
        print('Usage: python util/multi_render.py src.pickle dst.mp4 --sta start_indx --sto stop_indx --step step_size')
    else:
        optional = {}
        if '--sta' in sys.argv:
            optional['start_frame'] = sys.argv[sys.argv.index('--sta') + 1]
        if '--sto' in sys.argv:
            optional['stop_frame'] = sys.argv[sys.argv.index('--sto') + 1]
        if '--step' in sys.argv:
            optional['step'] = sys.argv[sys.argv.index('--step') + 1]
        block_file = sys.argv[1]
        out_file = sys.argv[2]
        if optional:
            main(block_file, out_file, optional)
        else:
            main(block_file, out_file)