import os
# Set non-random initial muscle activations
import sys
from abc import ABC, abstractmethod
from typing import Optional
import random

import gym
import numpy as np

# Add sconepy folders to path
if sys.platform.startswith("win"):
    sys.path.append("C:/Program Files/SCONE/bin")
elif sys.platform.startswith("linux"):
    sys.path.append("/opt/scone/lib")
elif sys.platform.startswith('darwin'):
    sys.path.append("/Applications/SCONE.app/Contents/MacOS/lib")

import sconepy

# 1) Template with raw string + escaped braces
scenario_template = r'''
ModelHyfydy {{
  model_file = C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\H2190v2k.hfd
  state_init_file = C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\InitStateGait10_uneven.zml
  fixed_control_step_size = 0.005
  initial_equilibration_activation = 0.01
  zero_velocity_equilibration = 1
  use_omnidirectional_root_dofs = 1
  use_opensim_activation_dynamics = 1
  {blueprint}
}}
'''

def generate_blueprint():
    slopes = [random.randint(-6, 6) for _ in range(50)]
    slopes_str = ' '.join(str(s) for s in slopes)
    return (
        'blueprint = {\n'
        '    path {\n'
        '        tiles = 50\n'
        '        pos = [ 0 5 0 ]\n'
        '        tile_dim = [ 1 0.1 4 ]\n'
        f'        slopes = [ {slopes_str} ]\n'
        '    }\n'
        '}\n'
    )

output_dir = r"C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\random_terrain"
for i in range(1, 101):
    blueprint = generate_blueprint()
    content = scenario_template.format(
        model_file=r'C:\Users\IlseungPark\Documents\scone_sota\sconegym-main\sconegym\data-v1\H2190v2k.hfd',
        blueprint=blueprint
    )
    path = os.path.join(output_dir, f'random_terrain_{i}.scone')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
print(f"Generated 3 random-terrain scenarios in {output_dir}")
