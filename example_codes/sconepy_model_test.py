import numpy as np
import time
from sconetools import sconepy
from sto2synergy_mat import generate_synergy_matrix

# ─── 사다리꼴 프로파일 정의 ────────────────────────────────────────────────
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

# ─── 시뮬레이션 함수 ─────────────────────────────────────────────────────
def run_simulation_synergy(model,
                           store_data,
                           H,               # synergy matrix (n_synergies × n_muscles)
                           muscle_cols,     # activation column names 리스트
                           synergy_index,   # 테스트할 시너지 번호
                           max_time=3.0,
                           dt=0.01,
                           min_com_height=0.3):
    model.reset()
    model.set_store_data(store_data)
    model.init_state_from_dofs()

    ## H 열 순서에 대응하는 muscle index 리스트 생성
    # 모델 전체 근육 이름
    muscle_names  = [m.name() for m in model.muscles()]
    
    # activation_cols → 실제 인덱스로 매핑(muscle_cols)
    valid_indices = [
        muscle_names.index(col.replace('/forceset/', '').replace('/activation', ''))
        for col in muscle_cols
    ]
    
    
    for t in np.arange(0, max_time + dt/2, dt):
        # 시너지 활성도 하나만 trapezoid_profile → muscle activations
        h_t = trapezoid_profile(t,
                                t_start=0.5,
                                t_rise=0.5,
                                t_plateau=4.0,
                                t_fall=0.5,
                                amplitude=1.0)
        # 50-길이 zero 벡터 생성 후 valid_indices 위치에만 시너지 활성도 할당
        ext = np.zeros(len(muscle_names), dtype=np.float32)
        ext[valid_indices] = (h_t * H[synergy_index]).astype(np.float32)

        model.set_actuator_inputs(ext)
        model.advance_simulation_to(t)

        if model.com_pos().y < min_com_height:
            print(f'[Synergy {synergy_index}] Abort at t={model.time():.3f}, com_y={model.com_pos().y:.3f}')
            break

    if store_data:
        dirname  = f'sconepy_synergy_{H.shape[0]}_{model.name()}'
        filename = f'{model.name()}_s{synergy_index}_{model.time():.3f}_{model.com_pos().y:.3f}'
        model.write_results(dirname, filename)
        print(f'[Synergy {synergy_index}] Results → {dirname}/{filename}')

# ─── 모든 시너지 테스트 ─────────────────────────────────────────────────
def batch_test_all_synergies(model_file,
                             data_path,
                             n_components=6,
                             exclude_muscles=None):
    # 1) 시너지 행렬 생성
    H, muscle_cols  = generate_synergy_matrix(data_path,
                                n_components=n_components,
                                exclude_muscles=exclude_muscles or [])

    # 2) 모델 로딩 및 SCONE 설정
    model = sconepy.load_model(model_file)
    sconepy.set_log_level(3)
    sconepy.set_array_dtype_float32()

    # 3) 각 시너지별 시뮬레이션
    for si in range(H.shape[0]):
        run_simulation_synergy(model,
                               store_data=True,
                               H=H,
                               muscle_cols=muscle_cols,
                               synergy_index=si,
                               max_time=8.0,
                               dt=0.01,
                               min_com_height=-10)

# ─── 메인 ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    MODEL_FILE = r'sconegym\data-v1\H2190_modeltest.scone'
    DATA_PATH  = r'E:\Dropbox\walk_rl\Data\mocap\MocoInv\H2190\BeforeTendonCompliance_R'
    N_SYNERGY  = list(range(4,13))  # 4~12개 시너지 테스트
    EXCLUDE    = ['rect_abd_r', 'ext_obl_r', 'int_obl_r', 'quad_lumb_r', 'erec_sp_r']

    for N_SYN in N_SYNERGY:
        batch_test_all_synergies(MODEL_FILE,
                                DATA_PATH,
                                n_components=N_SYN,
                                exclude_muscles=EXCLUDE)
