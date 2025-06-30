import os
from gym.envs.registration import register
curr_dir = os.path.dirname(os.path.abspath(__file__))


# Walk Environments

register(id="sconewalk_h2190-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data-v1/H2190.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10, 11],
             'right_leg_idxs': [12, 13, 14, 15, 16, 17],
             'clip_actions': False,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": 0,
                # "joint_limit_coeff": -0.1307,
                "joint_limit_coeff": -1,
                "smooth_coeff": 0,
                "nmuscle_coeff": 0,
                "self_contact_coeff": 0.0,
             }
         },
        )

register(id="sconewalk_h2190_uneven6-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data-v1/random_terrain_6.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10, 11],
             'right_leg_idxs': [12, 13, 14, 15, 16, 17],
             'clip_actions': False,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": 0,
                # "joint_limit_coeff": -0.1307,
                "joint_limit_coeff": -1,
                "smooth_coeff": 0,
                "nmuscle_coeff": 0,
                "self_contact_coeff": 0.0,
             }
         },
        )

