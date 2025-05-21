import numpy as np
import time
from sconetools import sconepy

def trapezoid_profile(t,
                      t_start=0.5,    # 비활성 구간 길이 [s]
                      t_rise=0.5,     # 상승 구간 길이 [s]
                      t_plateau=1.0,  # 유지 구간 길이 [s]
                      t_fall=0.5,     # 하강 구간 길이 [s]
                      amplitude=1.0): # 최대 활성화
    """
    t < t_start              : 0
    t_start <= t < t_start+t_rise
                             : 선형 상승 (0→amplitude)
    t_start+t_rise <= t < t_start+t_rise+t_plateau
                             : amplitude 유지
    t_start+t_rise+t_plateau <= t < ...+t_fall
                             : 선형 하강 (amplitude→0)
    그 외                    : 0
    """
    if t < t_start:
        return 0.0
    elif t < t_start + t_rise:
        return amplitude * (t - t_start) / t_rise
    elif t < t_start + t_rise + t_plateau:
        return amplitude
    elif t < t_start + t_rise + t_plateau + t_fall:
        return amplitude * (1 - (t - (t_start + t_rise + t_plateau)) / t_fall)
    else:
        return 0.0

def run_simulation(model, store_data, muscle_index,
                   max_time=3.0, dt=0.01, min_com_height=0.3):
    model.reset()
    model.set_store_data(store_data)
    model.init_state_from_dofs()

    n_mus = len(model.muscles())
    for t in np.arange(0, max_time + dt/2, dt):
        # 모든 근육 0, 특정 인덱스만 trapezoid_profile
        ext = np.zeros(n_mus, dtype=np.float32)
        ext[muscle_index] = trapezoid_profile(
            t,
            t_start=0.5,
            t_rise=0.5,
            t_plateau=1.0,
            t_fall=0.5,
            amplitude=1.0
        )
        model.set_actuator_inputs(ext)
        model.advance_simulation_to(t)

        if model.com_pos().y < min_com_height:
            print(f'[Muscle {muscle_index}] Abort at t={model.time():.3f}, com_y={model.com_pos().y:.3f}')
            break

    if store_data:
        dirname  = f'sconepy_muscle_{model.name()}'
        filename = f'{model.name()}_m{muscle_index}_{model.time():.3f}_{model.com_pos().y:.3f}'
        model.write_results(dirname, filename)
        print(f'[Muscle {muscle_index}] Results → {dirname}/{filename}')

def batch_test_all_muscles(model_file):
    model = sconepy.load_model(model_file)
    sconepy.set_log_level(3)
    sconepy.set_array_dtype_float32()

    for mi in range(len(model.muscles())):
        run_simulation(model, store_data=True, muscle_index=mi,
                       max_time=3.0, dt=0.01, min_com_height=-10)

if __name__ == '__main__':
    batch_test_all_muscles('sconegym\\data-v1\\H2190_modeltest.scone')
