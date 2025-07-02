import gym
import numpy as np
from gym import Wrapper, spaces
from gym.spaces import Box

class H2190Wrapper(Wrapper):
    """
    Wrapper that:
      1) env.init_activations_mean/std 자동 설정
      2) action → [-1,1]→[0,1] 매핑
      3) (옵션) synergy 차원 → full actuator 차원 변환
      4) (옵션) phase(phi) ≥ 0.5 시 left/right action 교환 (symmetry)
      5) _get_obs_3d 수정본 로직 적용
    """
    def __init__(
        self,
        env: gym.Env,
        syn_matrix: np.ndarray, 
        n_syn: int,
        use_synergy: bool = True,
        use_symmetry: bool = True,
        init_activations_mean: float = 0.01,
        init_activations_std:  float = 0.0,
        fall_penalty: float = 100.0,
        step_size: float = 0.025
    ):
        super().__init__(env)
        # ── phase_detect용 초기화 ──
        self.time = 0.0
        self.phase = 0.0
        self.last_event_time = 0.0
        self.prev_left_contact  = False
        self.prev_right_contact = False
        self.last_strike = None

        # 1) 초기 활성화 파라미터 설정
        env.init_activations_mean = init_activations_mean
        env.init_activations_std  = init_activations_std
        self._do_store_next = False
        # synergy/symmetry 플래그 & 매트릭스
        self.syn_matrix   = syn_matrix
        self.n_syn        = n_syn
        self.use_synergy  = use_synergy
        self.use_symmetry = use_symmetry
        self.fall_penalty = fall_penalty
        self.step_size = step_size

        # action_space 재정의
        act_dim = 2 * n_syn+10 if use_synergy else env.action_space.shape[0]
        self.action_space = spaces.Box(
            low  = -np.ones((act_dim,)),
            high =  np.ones((act_dim,)),
            dtype=np.float32
        )
        # observation_space는 원본 그대로
        dummy_obs = self._get_obs_3d()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

    def phase_detect(self):
        """
        Event-based phase detection with clipping:
        - Left heel-strike -> phase = 0.0, 'left' region
        - Right heel-strike -> phase = 0.5, 'right' region
        Then advance phase by elapsed time over half-cycle duration,
        and clip phase within the current half-cycle interval.
        """
        # Detect heel contacts
        Rleg = self.env.model.legs()[1]
        Lleg = self.env.model.legs()[0]
        
        left_contact  = (Lleg.contact_force().y > 1)
        right_contact = (Rleg.contact_force().y > 1)

        # Current simulation time
        current_time = self.time

        # Reset phase and region on heel-strike events
        if right_contact and not self.prev_right_contact:
            self.phase = 0.0
            self.last_event_time = current_time
            self.last_strike = 'right'
        elif left_contact and not self.prev_left_contact:
            self.phase = 0.5
            self.last_event_time = current_time
            self.last_strike = 'left'

        # Update previous contact states
        self.prev_right_contact = right_contact
        self.prev_left_contact  = left_contact

        # Advance phase based on elapsed time
        elapsed = current_time - self.last_event_time

        # linVL 모델 계수 계산
        # 입력: 신장과 체중
        height_cm = 175.0
        weight_kg = 74.5314

        # 키를 m 단위로 변환하고 BMI 계산
        height_m = height_cm / 100
        BMI = weight_kg / (height_m ** 2)
        # 나이를 본인의 나이로 설정하세요
        age = 30  # 예: 30세

        # linVL 모델 계수 계산
        c1 = 0.2551 - 0.0057 * BMI
        c3 = 1.6951 + 0.0031 * age - 0.0242 * BMI

        v= self.env.model.com_vel().x
        v=np.clip(v, 0.1, 2)  # 속도 범위 제한 (0.1 m/s ~ 2.0 m/s)

        gait_cycle_duration = c1 / v + c3

        self.phase = self.phase + elapsed / gait_cycle_duration

        # Clip phase to the current half-cycle interval
        if self.last_strike == 'right':
            self.phase = max(0.0, min(self.phase, 0.5))
        elif self.last_strike == 'left':
            self.phase = max(0.5, min(self.phase, 1.0))

        return float(self.phase)
    

    def reset(self, **kwargs):
        
        # 멀쩡히 리셋 호출
        self.env.model.set_store_data(self._do_store_next)
        if self._do_store_next:
            self.env.store_next_episode()
            self._do_store_next = False
        _ = self.env.reset(**kwargs)
        self.time = 0.0
        self.phase = 0.0
        self.last_event_time = self.env.time    # super().reset() 후 time은 0.0
        self.prev_left_contact  = False
        self.prev_right_contact = False
        self.last_strike = None
        return self._get_obs_3d()

    def step(self, action):
        # 2) [-1,1] → [0,1]
        self.env.steps += 1
        action = np.clip(action, -1.0, 1.0)
        action = 0.5 * (action + 1.0)

        # 3) synergy 적용
        if self.use_synergy:
            a_r_leg = action[:self.n_syn]
            a_l_leg = action[self.n_syn:2*self.n_syn]
            a_r_torso = action[2*self.n_syn:2*self.n_syn+5]
            a_l_torso = action[2*self.n_syn+5:2*self.n_syn+10]
            if self.use_symmetry:
                phi = self.phase_detect()
                if phi >= 0.5:
                    action = np.concatenate([a_l_leg.dot(self.syn_matrix), a_r_leg.dot(self.syn_matrix), a_l_torso, a_r_torso], axis=0)
                else:
                    action = np.concatenate([a_r_leg.dot(self.syn_matrix), a_l_leg.dot(self.syn_matrix), a_r_torso, a_l_torso], axis=0)
            else:
                action = np.concatenate([
                    a_r_leg.dot(self.syn_matrix),
                    a_l_leg.dot(self.syn_matrix),
                    a_r_torso,
                    a_l_torso
                ], axis =0)
        action = np.clip(action, 0, 1.0)

        self.env.model.set_actuator_inputs(action)
        self.env.model.advance_simulation_to(self.time + self.step_size)
        # 원본 step 실행 (obs 무시)
        # 6) compute reward with custom weighting
        rwd_dict = self._update_rwd_dict()
        reward   = sum(rwd_dict.values())
        obs = self._get_obs_3d()
        done   = self._get_done()
        reward = self.env._apply_termination_cost(reward, done)

        
        info = {
        "gaussian_vel": rwd_dict["gaussian_vel"],
        "constr"      : rwd_dict["constr"],
        "effort"      : rwd_dict["effort"],
        }
        if done:
            # env.steps 는 매 step 마다 GaitGym 내부에서 ++ 됩니다
            max_steps = getattr(self.env, '_max_episode_steps', None)
            if max_steps is not None and self.env.steps >= max_steps:
                # 시간 만료(terminal by timeout)
                info['timed_out'] = True
            else:
                # 중간에 넘어져서 떨어짐(terminal by fall)
                info['fell'] = True
                # fall 에만 패널티 적용
                reward -= self.fall_penalty
        # 8) update time & total reward
        self.time += self.step_size
        self.env.total_reward += reward

        # 9) end‐of‐episode bookkeeping
        if done:
            if self.env.store_next:
                self.env.model.write_results(
                    self.env.output_dir,
                    f"{self.env.episode:05d}_{self.env.total_reward:.3f}"
                )
                self.env.store_next = False
            self.env.episode += 1

        return obs, reward, done, info

    def mirror_obs_numpy(self, obs: np.ndarray) -> np.ndarray:
        """
        1D obs (shape [obs_dim], numpy) 를 좌우 대칭 뒤집어 반환합니다.
        """
        obs_m = obs.copy()

        # ── 1)~4) muscle 관련 4개 블록 (각각 90차원)
        mus = 90
        r1 = np.arange(0, 40)
        l1 = np.arange(40, 80)
        tr = np.arange(80, 85)
        tl = np.arange(85, 90)
        idx_m = np.concatenate([l1, r1, tl, tr])
        for b in range(4):
            s = b * mus
            obs_m[s : s + mus] = obs[s : s + mus][idx_m]

        # ── 5) head orientation (quaternion w,x,y,z) : x,y 성분 부호 반전
        head_ori_start = 4 * mus
        # w 성분 그대로
        obs_m[head_ori_start + 0] = obs[head_ori_start + 0]
        # x, y 성분은 반전
        obs_m[head_ori_start + 1] = -obs[head_ori_start + 1]
        obs_m[head_ori_start + 2] = -obs[head_ori_start + 2]
        # z 성분 그대로
        obs_m[head_ori_start + 3] = obs[head_ori_start + 3]

        # ── 6) head_angv (3차원) : [vx,vy,vz] -> [-vx, -vy, vz]
        h = head_ori_start + 4
        obs_m[h + 0] = -obs[h + 0]
        obs_m[h + 1] = -obs[h + 1]
        obs_m[h + 2] =  obs[h + 2]

        # ── 7) feet (6차원) : [0,1,2,3,4,5] -> [-3,4,5, -0,1,2]
        f = h + 3
        b0 = obs[f : f + 6]
        obs_m[f + 0] =  b0[3]
        obs_m[f + 1] =  b0[4]
        obs_m[f + 2] = -b0[5]
        obs_m[f + 3] =  b0[0]
        obs_m[f + 4] =  b0[1]
        obs_m[f + 5] = -b0[2]

        # ── 8) dof_values (21차원)
        d = f + 6
        idx2_m = np.array([0,1,2,3,4,5,12,13,14,15,16,17,6,7,8,9,10,11,18,19,20], dtype=int)
        signs_pos = np.array([ 1,-1,-1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1], dtype=obs.dtype)
        blk = obs[d : d + 21]
        perm = blk[idx2_m]
        obs_m[d : d + 21] = perm * signs_pos

        # ── 9) dof_vels (21차원)
        v = d + 21
        blk = obs[v : v + 21]
        perm = blk[idx2_m]
        signs_vel = np.array([ 1,-1,-1, 1, 1,-1, 1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1], dtype=obs.dtype)
        obs_m[v : v + 21] = perm * signs_vel

        # ── 10) acts (90차원) : muscle 과 동일하게 swap
        a = v + 21
        obs_m[a : a + mus] = obs[a : a + mus][idx_m]

        # ── 11) grf (6차원) : [0,1,2,3,4,5] -> [3,4,-5, 0,1,-2]
        g = a + mus
        b0 = obs[g : g + 6]
        obs_m[g + 0] =  b0[3]
        obs_m[g + 1] =  b0[4]
        obs_m[g + 2] = -b0[5]
        obs_m[g + 3] =  b0[0]
        obs_m[g + 4] =  b0[1]
        obs_m[g + 5] = -b0[2]

        et =g+6
        b0 = obs[et : et + 2]
        obs_m[et + 0] = b0[1]  # exo_torque_r
        obs_m[et + 1] = b0[0]  # exo_torque_l
        # ── 12) com_vel (3차원) : [vx,vy,vz] -> [vx, vy, -vz]
        c = et + 2
        b0 = obs[c : c + 3]
        obs_m[c + 0] =  b0[0]
        obs_m[c + 1] =  b0[1]
        obs_m[c + 2] = -b0[2]

        # tv(1)와 slope(1)은 그대로
        return obs_m
    
    def _get_obs_3d(self):
        # same as your modified method, but using self.env.*
        env = self.env

        acts = env.model.muscle_activation_array()
        env.prev_acts = acts.copy()
        env.prev_excs = env.model.muscle_excitation_array()
        
        Lleg = env.model.legs()[0]
        Rleg = env.model.legs()[1] 
        grf = np.array([ Lleg.contact_force().x , Lleg.contact_force().y , Lleg.contact_force().z,
                         Rleg.contact_force().x , Rleg.contact_force().y , Rleg.contact_force().z ]) / (env.model.mass()*9.81)
        com_vel = np.array([env.model.com_vel().x, env.model.com_vel().y, env.model.com_vel().z])
        dof_values = env.model.dof_position_array()
        dof_vels   = env.model.dof_velocity_array()
        dof_values[3] = 0.0
        dof_values[4] = 0.0
        dof_values[5] = 0.0

        m_fibl = env.model.muscle_fiber_length_array()
        m_fibv = env.model.muscle_fiber_velocity_array()
        m_force= env.model.muscle_force_array()
        m_exc  = env.model.muscle_excitation_array()

        head_or  = env.head_body.orientation().array()
        # head_acc = env.head_body.com_acc().array()
        head_angv= env.head_body.ang_vel().array()
        feet     = env._get_feet_relative_position()
        phi = self.phase_detect()
        
        raw_obs = np.concatenate([
            m_fibl, m_fibv, m_force, m_exc,
           head_or, head_angv,
            feet, dof_values, dof_vels, acts, grf, com_vel
        ], dtype=np.float32).copy()

        if self.use_symmetry ==True:
            phi = self.phase_detect()
            if phi >= 0.5:
                raw_obs = self.mirror_obs_numpy(raw_obs)


        
        return raw_obs
    
        
    def _update_rwd_dict(self):
        # here self is the wrapper, so use self.env inside
        env = self.env
        return {
            "gaussian_vel": env.vel_coeff      * self._gaussian_vel(),
            "grf"         : 0            * env._grf(),
            "smooth"      : 0   * env._exc_smooth_cost(),
            "number_muscles": 0* env._number_muscle_cost(),
            "constr"      : env.joint_limit_coeff * env._joint_limit_torques(),
            "self_contact": 0  * env._get_self_contact(),
            "effort"      : -0.1 * self._effort_cost(),
            
        }
    
    def _gaussian_vel(self):
        # your new gaussian_vel implementation
        env = self.env
        c = 0.06
        v_var_x = 0.07**2
        v_var_z = 0.15**2
        lumb_rot_var=0.2**2
        ori_var = 0.06**2
        # angvel_var = [0.2442**2, 0.2603**2, 0.3258**2] %너무 강해서 제거함함
        angvel_var = [0.6**2, 0.65**2, 1.2**2]
        acc_var    = [0.4799**2, 1.7942**2, 0.7716**2]

        vel_x = env.model_velocity()
        vel_z = env.model.com_vel().z
        lumb_rot = env.model.dof_position_array()[-1]
        head_angvel = env.head_body.ang_vel().array()
        head_acc    = env.head_body.com_acc().array()
        head_ori = env.head_body.orientation().y
        
        # vel_x는 목표 속도보다 작을 때는 gaussian 보상
        if vel_x < env.target_vel:
            r_vel_x = np.exp(-c * (vel_x - env.target_vel)**2 / v_var_x)
        else:
        # 목표 속도보다 크면 최대 보상인 1로 plateau
            r_vel_x = 1

        
        r_vel_z = np.exp(-c * vel_z**2 / v_var_z)
        # r_lumb_rot= np.exp(-c * lumb_rot**2 / lumb_rot_var)
        r_ori = np.exp(-c * head_ori**2 / ori_var)
        #----------아래는 3축다!------------
        r_head_angvel = np.prod([
            np.exp(-c * head_angvel[i]**2 / angvel_var[i])
            for i in range(0,3)
        ])


        #----------------------------------
        #----------아래는 1축만--------------
        # r_head_angvel = np.prod([
        #     np.exp(-c * head_angvel[1]**2 / angvel_var[1])
        # ])
        r_head_acc = np.prod([
            np.exp(-c * head_acc[i]**2 / acc_var[i])
            for i in range(3)
        ])
        #----------------------------------
        return r_vel_x * r_head_angvel*r_vel_z
    

    def _effort_cost(self):
        """
        Computes the normalized effort cost based on the cubic muscle activations
        plus scaled torso actuator inputs.
        """
        env = self.env
        # muscle activations
        activations = env.model.muscle_activation_array()

        return np.sum(np.power(activations, 3))
    
    def _get_done(self) -> bool:
            """
            The episode ends if the center of mass is below min_com_height.
            """
            env= self.env
            # COM 절대 높이 대신, 가장 낮은 발 위치를 기준으로 상대 높이 계산
            com_y    = env.model.com_pos().y
            foot_l_y = (
                [y for y in env.model.bodies() if "calcn_l" in y.name()][0]
                .com_pos().y
            )
            foot_r_y = (
                [y for y in env.model.bodies() if "calcn_r" in y.name()][0]
                .com_pos().y
            )
            ground_y = min(foot_l_y, foot_r_y)

            # 골반 기준 상대 높이가 threshold 이하이면 '넘어진 것'으로 간주
            REL_HEIGHT_THRESHOLD = 0.3  # 예시값: 0.2m
            if com_y - ground_y < -REL_HEIGHT_THRESHOLD:
                return True
            if com_y - ground_y < REL_HEIGHT_THRESHOLD:
                return True
            if env.steps >= env._max_episode_steps:
                return True
            return False
    
    def store_next_episode(self):
        # wrapper 레벨에서 플래그만 세팅
        self._do_store_next = True

    def write_now(self):
        return self.env.write_now()
