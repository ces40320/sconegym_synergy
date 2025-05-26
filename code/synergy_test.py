import numpy as np
import random
import time
# sconetools -- helper module to find sconepy
# (C) Copyright Thomas Geijtenbeek
# This file is part of SCONE. For more information, see http://scone.software.
import sys
import os
import platform
import pathlib
import importlib.util

# if sys.version_info < (3,9) or sys.version_info >= (3,10):
#     raise Exception("sconepy only supports Python 3.9 -- current version: " + platform.python_version() )

path_to_sconepy = ""

def try_find_sconepy(pathlist):
    global path_to_sconepy
    if path_to_sconepy:
        return; # already found
    for path in pathlist:
        if path:
            if sorted(pathlib.Path(path).glob("sconepy*.*")):
                path_to_sconepy = str(path)
                return

# search for sconepy in os-specific paths
if importlib.util.find_spec("sconepy") is None:
    path_list = []

    if sys.platform.startswith("win"):
        if scone_install := os.getenv("SCONE_PATH"):
            path_list.append(scone_install + "/bin")
        path_list.extend([
            os.getenv("LOCALAPPDATA") + "/SCONE/bin",
            os.getenv("ProgramFiles") + "/SCONE/bin",
            ])

    # find sconepy in path_list
    try_find_sconepy(path_list)

    # check if we succeeded
    if path_to_sconepy:
        print("Found sconepy at", path_to_sconepy)
        sys.path.append(path_to_sconepy)
    else:
        print("Could not find sconepy in", path_list)
        raise Exception("Could not find SconePy in " + path_list)

import sconepy

from sconepy import  Geometry

# 모델 로드
model = sconepy.load_model("models/H1922v2n.hfd")  # 기존 HFD 모델 :contentReference[oaicite:7]{index=7}

# 랜덤한 박스 지형 10개 추가
for i in range(10):
    x_pos   = (i + 1) * random.uniform(0.5, 1.0)      # 박스 간 x 간격 랜덤 :contentReference[oaicite:8]{index=8}
    height  = random.uniform(0.01, 0.1)               # 높이 랜덤
    box = Geometry(
        name            = f"terrain_box_{i}",
        body            = "ground",
        type            = "box",
        pos             = Vec3(x_pos, height/2, 0),
        dim             = Vec3(1.0, height, 1.0),
        ori             = Vec3(0, 0, random.uniform(-15, 15))
    )
    model.add_geometry(box)

# 새로운 HFD 파일로 저장
model.save("models/H1922_uneven.hfd")               # 변경된 모델 저장 :contentReference[oaicite:9]{index=9}

# SCONE 시나리오 업데이트
scenario = sconepy.Scenario.from_file("scenarios/gait.scone")
scenario.model.file = "models/H1922_uneven.hfd"
