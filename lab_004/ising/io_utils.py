import numpy as np
from pathlib import Path

def save_magnetization_data(filename, magnetization):
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    data = np.column_stack((np.arange(len(magnetization)), magnetization))
    np.savetxt(filepath, data, fmt=['%d', '%.6f'], delimiter=',', header='step,magnetization', comments='')


def save_animation(anim, filename):
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    writer = 'ffmpeg' if filepath.suffix == '.mp4' else 'pillow'
    print(f'SAVING ANIMATION TO {filepath}...')
    anim.save(filepath, writer=writer)