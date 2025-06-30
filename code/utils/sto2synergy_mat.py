import glob
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import NMF

# 파일 경로 설정
# data_path = r'E:\Dropbox\walk_rl\Data\mocap\MocoInv\R'
# n_components = 12  # 차원 축소 후의 차원 수

def generate_synergy_matrix(data_path, n_components = 6, exclude_muscles=False, include_muscles = False):
    """
    지정 경로의 .sto 파일들을 읽어들여 활성화 데이터 행렬을 구성하고,
    NMF를 통해 n_components 차원으로 분해한 후 H 행렬과 해당 의사역행렬(pseudo-inverse)을 계산하여 반환합니다.
    
    Parameters:
        data_path (str): .sto 파일들이 있는 디렉토리 경로
        n_components (int): NMF 분해 후 원하는 차원 수 (컴포넌트 수)
    
    Returns:
        H (numpy.ndarray): 분해 후의 H 행렬 (n_components x features)
        pseudo_inverse (numpy.ndarray): H 행렬의 모어-펜로즈 의사역행렬
        explained_variance (float): 재구성을 통해 설명된 분산 비율
    """
    # glob 모듈을 사용하여 .sto 파일 리스트 불러오기
    sto_files = glob.glob(f"{data_path}/*.sto")

    # 모든 파일 데이터를 저장할 빈 리스트 생성
    activation_ndarray = []

    for idx, file in enumerate(sto_files):
        # 파일을 읽고 헤더와 데이터를 분리하여 로드
        with open(file, 'r') as f:
            lines = f.readlines()

        # 헤더의 끝을 찾아서 데이터 시작 줄을 확인
        header_end_line = next(i for i, line in enumerate(lines) if 'endheader' in line)

        # pandas로 데이터 읽기
        df = pd.read_csv(file, sep='\t', skiprows=header_end_line+1)

        # 마지막에 '_r/activation'으로 끝나는 열만 추출
        activation_cols = [col for col in df.columns if col.endswith('_r/activation')]
        
        if exclude_muscles is not False:
            # 상체근육 제외 ['rect_abd_r', 'ext_obl_r', 'int_obl_r', 'quad_lumb_r', 'erec_sp_r']
            activation_cols = [col for col in activation_cols if not any(exclude in col for exclude in exclude_muscles)]
        elif include_muscles is not False:
            # 특정 근육만 포함
            activation_cols = [col for col in activation_cols if any(include in col for include in include_muscles)] 
            
        
        activation_df = df[activation_cols]

        # 처음과 끝에서 8개의 데이터(0.04초 분량) 제외
        activation_data_trimmed = activation_df.iloc[8:-8]

        # 데이터를 numpy로 변환하고 기존 ndarray에 추가
        if not isinstance(locals().get('activation_ndarray', None), np.ndarray):
            activation_ndarray = activation_data_trimmed.to_numpy()
        else:
            activation_ndarray = np.vstack((activation_ndarray, activation_data_trimmed.to_numpy()))

    # 최종 데이터 확인
    print("activation_ndarray shape:", activation_ndarray.shape)


    # NMF 모델 생성 및 적용

    nmf_model = NMF(n_components=n_components, init='random', random_state=1000000, max_iter=100000000)

    W = nmf_model.fit_transform(activation_ndarray)
    H = nmf_model.components_


    for i in range(0, n_components):
        max_value = np.max(W[:, i])

        W[:, i] /= max_value
        H[i, :] *= max_value

    for i in range(0, n_components):
        print(f"Maximum W[{i},:] = ", np.max(W[:,i]))
    
    for i in range(0, n_components):
        print(f"Maximum H[{i},:] = ", np.max(H[i,:]))
        

    return H, activation_cols




if __name__ == "__main__":
    # 데이터 경로와 차원 수를 지정하여 함수 호출
    # generate_synergy_matrix(data_path, n_components)

    data_path = r'E:\Dropbox\walk_rl\Data\mocap\MocoInv\H2190\BeforeTendonCompliance_R' # 24_2\Biomechanical Analysis of Motor Skill\
    n_components = 4  # 차원 축소 후의 차원 수
    selected_muscles =  ['rect_abd_r', 'ext_obl_r', 'int_obl_r', 'quad_lumb_r', 'erec_sp_r'] # 상체 근육 제외
    
    H, _ = generate_synergy_matrix(data_path, n_components, exclude_muscles=selected_muscles)   # include_muscles 인지 exclude_muscles 인지 확인 필수

    # H 행렬 출력
    print("H = np.array([")
    for row in H:
        print("    [", ", ".join(f"{value:.8e}" for value in row), "],")
    print("])")

    # 의사역행렬 (Moore-Penrose pseudo-inverse) 계산 및 출력
    pseudo_inverse = np.linalg.pinv(H)
    print("pseudo_inverse = np.array([")
    for row in pseudo_inverse:
        print("    [", ", ".join(f"{value:.8f}" for value in row), "],")
    print("])")