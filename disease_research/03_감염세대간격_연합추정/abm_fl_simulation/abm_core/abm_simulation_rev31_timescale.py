# abm_simulation_integrated.py 최상단 코드 확인
import sys
import os
import glob

# -*- coding: utf-8 -*-

# ==============================================================================
# ==============================================================================
# ===== SECTION 0: Imports and Setup ===========================================
# ==============================================================================
# ==============================================================================

import os
import xml.etree.ElementTree as ET
import random
import numpy as np
import warnings
import csv
import pandas as pd
from xml.dom import minidom
import time
from datetime import datetime
import traceback
import platform # [신규] platform 모듈 임포트

# Optional: Used in rate_generator.py
try:
    from scipy.optimize import curve_fit
except ImportError:
    print("Warning: scipy not found. Rate generation functions might be limited.")
    curve_fit = None

# Optional: Used for visualization in reporting.py
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    plt = None
    print("Warning: matplotlib not found. Graph generation will be disabled.")

# ==============================================================================
# ==============================================================================
# ===== SECTION 1: Constants (constants.py 통합 및 수정) =======================
# ==============================================================================
# ==============================================================================

# (이 섹션은 변경 없음)

# --- 상태 및 유형을 정수로 매핑 ---
# S: Susceptible, E: Exposed, I: Infectious (Mild), H: Hospitalized, R: Recovered, D: Deceased
STATUS_MAP = {'S': 0, 'E': 1, 'I': 2, 'H': 3, 'R': 4, 'D': 5}

# AGENT_TYPE_MAP (수정: 미취학 아동 추가, 비경제활동 추가)
AGENT_TYPE_MAP = {
    'worker': 0,  # 근로자
    'preschooler': 1,  # 미취학 아동 (3-7세)
    'student_e': 2,  # 초등학생
    'student_mh': 3,  # 중고등학생
    'senior': 4,  # 노년층
    'unemployed': 5  # 비경제활동인구 (목적지가 여가인 그룹)
}

MAX_AGE = 101

# 기본 중증화 발병 시점
DEFAULT_SEVERITY_ONSET_MIN = 7.0
DEFAULT_SEVERITY_ONSET_MAX = 11.0

# 시뮬레이션 시간 단위 ([복원 #14] 4-step 모델로 확장: 1일 = 4스텝)
TIME_STEP = 1 / 4.0

# 장소 유형 매핑 ([복원 #15] transport 활동 추가)
# 0: 집, 1: 주 목적지(직장/학교/유치원/경로당), 2: 여가(비경제활동), 3: 교통수단 (출근/통학)
PLACE_TYPE_MAP = {
    'home': 0,
    'destination': 1,
    'leisure': 2,
    'transport': 3,
}

# 감염 발생 위치 추적용 장소 유형 매핑 (수정: 유치원/어린이집 추가, 교통수단 제거)
INFECTION_PLACE_TYPE_MAP = {
    'household': 0,
    'workplace': 1,
    # 학교 유형 (2)
    'school': 2,
    'elementary_school': 2,
    'middle_high_school': 2,
    # 유치원/어린이집/유보통합기관 (3)
    'preschool': 3,
    'kindergarten': 3,
    'daycare': 3,
    'early_childhood_center': 3,
    'senior_center': 4,
    # 여가 유형 (5)
    'leisure': 5,
    'leisure_park_nature': 5,
    'leisure_cafe_restaurant': 5,
    'leisure_mall_store': 5,
    'leisure_entertainment': 5,
    'leisure_sports_facility': 5,
    'public': 6,  # 공공시설
    'other': 7,  # 기타/미정의
    # [복원 #15] 교통수단
    'transport': 8,
    'travelling': 8,  # schedules_bak.xml의 동의어
}

# 감염 위치 로깅 헤더 (수정 + [복원 #15] Inf_Transport)
INFECTION_PLACE_HEADERS = {
    0: 'Inf_Home', 1: 'Inf_Work', 2: 'Inf_School', 3: 'Inf_Preschool', 4: 'Inf_Senior',
    5: 'Inf_Leisure', 6: 'Inf_Public', 7: 'Inf_Other', 8: 'Inf_Transport',
}

# [신규] 여가(Leisure) 관련 타입들을 집합으로 정의 (이동확대 대상지 식별용)
LEISURE_TYPES = {k for k, v in INFECTION_PLACE_TYPE_MAP.items() if v == INFECTION_PLACE_TYPE_MAP['leisure']}

# 연령대 정의 (Scheduler에서 사용) (수정: 미취학 연령 추가)
# 참고: 이전 place_synthesizer 수정 시 3-7세로 정의되었으므로 이를 따름 (8세는 초등학생)
PRESCHOOL_AGE_MIN, PRESCHOOL_AGE_MAX = 3, 7
ELEMENTARY_AGE_MIN, ELEMENTARY_AGE_MAX = 8, 13
MIDDLE_HIGH_AGE_MIN, MIDDLE_HIGH_AGE_MAX = 14, 19
WORK_AGE_MIN, WORK_AGE_MAX = 20, 64
SENIOR_AGE_MIN = 65


# ==============================================================================
# ==============================================================================
# ===== SECTION 2: Scheduler (scheduler.py 통합 및 수정) =======================
# ==============================================================================
# ==============================================================================

# (이 섹션은 변경 없음)

class Scheduler:
    def __init__(self, schedules_file=None):
        """
        Scheduler 클래스를 초기화합니다.
        [복원 #14] 4-step 모델: 집(0) - 교통(1) - 목적지/여가(2) - 집(3).
        """
        self.simplified_schedule = {
            0: 'home',
            1: 'transport',     # 출근/통학
            2: 'destination',
            3: 'home',
        }

    # _load_schedules 함수 제거됨 (XML 미사용)

    def get_agent_type(self, agent_element):
        """
        에이전트의 속성을 기반으로 유형(역할)을 결정. (수정: 미취학 아동 분류 추가)
        """
        try:
            age = int(agent_element.get('age'))
        except (ValueError, TypeError):
            return 'unemployed'

        # 1. 미취학 아동 (신규)
        # 이전 단계에서 생성된 preschool_id 속성을 기반으로 판별
        if PRESCHOOL_AGE_MIN <= age <= PRESCHOOL_AGE_MAX and agent_element.get('preschool_id'):
            return 'preschooler'

        # 2. 학생 (초등/중고등)
        if agent_element.get('school_id'):
            if ELEMENTARY_AGE_MIN <= age <= ELEMENTARY_AGE_MAX:
                return 'student_e'
            elif MIDDLE_HIGH_AGE_MIN <= age <= MIDDLE_HIGH_AGE_MAX:
                return 'student_mh'

        # 3. 근로자
        if WORK_AGE_MIN <= age <= WORK_AGE_MAX and agent_element.get('workplace_id'):
            return 'worker'

        # 4. 노년층
        if age >= SENIOR_AGE_MIN and agent_element.get('senior_center_id'):
            return 'senior'

        # 그 외 (비경제활동인구)
        return 'unemployed'

    def get_activity(self, agent_type, step):
        """
        특정 에이전트 유형의 특정 스텝 활동을 반환. (Step 기반)
        [복원 #14] 4-step + 비경제활동인구는 transport 안함, destination 대신 leisure.
        """
        activity = self.simplified_schedule.get(step, 'home')

        if agent_type == 'unemployed':
            if activity == 'transport':
                return 'home'       # 비경제활동인구는 출퇴근 안함
            if activity == 'destination':
                return 'leisure'    # 외출

        return activity


# --- 스케줄러 더미 클래스 (Scheduler 로드 실패 시 대비 - 단순화 모델에서는 사용될 확률 낮음) ---
class DummyScheduler:
    def get_agent_type(self, agent_elem):
        # 최소한의 유형 분류 로직 (미취학 아동 포함)
        if agent_elem.get('preschool_id'): return 'preschooler'
        if agent_elem.get('workplace_id'): return 'worker'
        if agent_elem.get('school_id'):
            try:
                age = int(agent_elem.get('age'))
                if 8 <= age <= 13:
                    return 'student_e'
                elif 14 <= age <= 19:
                    return 'student_mh'
            except:
                pass
            return 'student_mh'
        if agent_elem.get('senior_center_id'): return 'senior'
        return 'unemployed'

    def get_activity(self, type_str, step):
        # [복원 #14] 4-step 더미 활동: 집(0) - 교통(1) - 목적지/여가(2) - 집(3)
        if step == 1:  # transport
            if type_str == 'unemployed':
                return 'home'
            return 'transport'
        if step == 2:  # destination/leisure
            if type_str == 'unemployed':
                return 'leisure'
            return 'destination'
        return 'home'  # step 0, 3


# ==============================================================================
# ==============================================================================
# ===== SECTION 3: Configuration Loader (config_loader.py 통합 및 수정) =========
# ==============================================================================
# ==============================================================================

# --- 스케줄러 로딩 및 준비 ---

def load_scheduler(schedules_file):
    """스케줄러를 로드합니다. 단순화 모델에서는 XML 파일 경로와 무관하게 항상 성공합니다."""
    # 단순화 모델에서는 항상 Scheduler 인스턴스를 성공적으로 반환
    return Scheduler()


def prepare_schedule_map(scheduler, cp):
    """스케줄 맵을 생성하여 GPU/CPU 메모리에 로드합니다. ([복원 #14] 4 스텝)"""
    num_agent_types = len(AGENT_TYPE_MAP)
    schedule_map_np = np.zeros((num_agent_types, 4), dtype=np.int8)
    if hasattr(scheduler, 'get_activity'):
        for type_str, type_int in AGENT_TYPE_MAP.items():
            for step in range(4):
                activity_str = scheduler.get_activity(type_str, step)
                schedule_map_np[type_int, step] = PLACE_TYPE_MAP.get(activity_str, 0)
    return cp.array(schedule_map_np)


# --- [신규] 이동확대 모드 설정 로딩 ---
# (원본 코드와 동일 - 이동확대 모드는 단순화 모델에서도 유지)
def load_travel_config(file_path):
    """
    travel_config.xml 파일을 로드하여 이동확대 스케줄과 목적지 정보를 파싱합니다.
    """
    travel_schedule = []
    destinations = []

    if not file_path:
        return {'schedule': travel_schedule, 'destinations': destinations}

    # 파일 경로가 절대 경로가 아니고 존재하지 않는 경우, 기본 경로(./inputs/)를 가정
    if not os.path.exists(file_path) and not os.path.isabs(file_path):
        file_path = os.path.join('./inputs', file_path)

    if not os.path.exists(file_path):
        # 파일이 없으면 경고 없이 조용히 넘어감 (이동확대 모드 비활성화)
        # warnings.warn(f"이동확대  설정 파일 '{file_path}'를 찾을 수 없습니다. 이동확대 모드를 사용하지 않습니다.")
        return {'schedule': travel_schedule, 'destinations': destinations}

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print(f"이동확대 설정 파일 로드: {file_path}")

        # 1. 스케줄 로드
        schedule_node = root.find("Schedule")
        if schedule_node is not None:
            for mode_node in schedule_node.findall("Mode"):
                try:
                    mode = {
                        'name': mode_node.get('name', 'UnnamedMode'),
                        'start_day': int(mode_node.get('start_day')),
                        # end_day는 포함됨 (예: start=1, end=1 이면 1일 하루동안 적용)
                        'end_day': int(mode_node.get('end_day')),
                        'override_probability': float(mode_node.get('override_probability', 0.5))
                    }
                    if mode['start_day'] <= mode['end_day'] and 0.0 <= mode['override_probability'] <= 1.0:
                        travel_schedule.append(mode)
                    else:
                        warnings.warn(f"이동확대 모드 '{mode['name']}'의 설정값이 유효 범위를 벗어났습니다.")
                except (ValueError, TypeError):
                    warnings.warn("이동확대 스케줄의 속성이 유효하지 않습니다. 해당 모드를 무시합니다.")

        # 2. 목적지(시군구) 로드
        destinations_node = root.find("Destinations")
        if destinations_node is not None:
            for dest_node in destinations_node.findall("Destination"):
                try:
                    sgg_id = int(dest_node.get('sigungu_id'))
                    # weight는 현재 버전에서는 사용되지 않으나 확장을 위해 파싱
                    weight = float(dest_node.get('weight', 1.0))
                    destinations.append({'sigungu_id': sgg_id, 'weight': weight})
                except (ValueError, TypeError):
                    warnings.warn("이동확대 대상지의 속성(sigungu_id)이 유효하지 않습니다.")

    except (ET.ParseError, FileNotFoundError) as e:
        warnings.warn(f"이동확대 설정 파일 '{file_path}' 처리 중 오류 발생({e}). 이동확대 모드를 사용하지 않습니다.")
        return {'schedule': [], 'destinations': []}

    # 스케줄을 시작일 기준으로 정렬
    travel_schedule.sort(key=lambda x: x['start_day'])

    print(f"로드된 이동확대 스케줄: {len(travel_schedule)}건, 이동확대 목적지 시군구: {len(destinations)}곳")
    return {'schedule': travel_schedule, 'destinations': destinations}


# --- 기본 데이터 로딩 (에이전트, 장소) 및 이동확대 대상지 준비 ---

def load_and_prepare_data(config, scheduler, cp):
    """
    에이전트, 장소 데이터를 로드하고 시뮬레이션 배열을 초기화합니다.
    [수정] preschool_id를 로드하고 transport_id를 제거합니다.
    """
    print("데이터 로드 및 배열 변환 중...")
    try:
        # 파일 경로는 config에 정의된 대로 사용 (run_simulation 함수 참조)
        agents_root = ET.parse(config['agents_file']).getroot()
        places_root = ET.parse(config['places_file']).getroot()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"필수 데이터 파일({e.filename})을 찾을 수 없습니다. (경로: {config['agents_file']} 또는 {config['places_file']})")
    except ET.ParseError as e:
        raise ValueError(f"XML 파일 파싱 오류: {e}")

    # ID 매핑 및 개수 확인
    agent_id_map = {elem.get('id'): i for i, elem in enumerate(agents_root.findall('agent'))}
    place_id_map = {elem.get('id'): i for i, elem in enumerate(places_root.findall('place'))}
    num_agents = len(agent_id_map)
    num_places = len(place_id_map)

    # === 1. 에이전트 데이터 파싱 ===
    # 에이전트 속성 배열 초기화 (NumPy 사용) (수정)
    agent_household_ids_np = np.full(num_agents, -1, dtype=np.int32)
    agent_workplace_ids_np = np.full(num_agents, -1, dtype=np.int32)
    agent_preschool_ids_np = np.full(num_agents, -1, dtype=np.int32)  # 신규: 유치원/어린이집 ID
    agent_school_ids_np = np.full(num_agents, -1, dtype=np.int32)
    agent_senior_center_ids_np = np.full(num_agents, -1, dtype=np.int32)
    # [복원 #15] agent_transport_ids_np 복원
    agent_transport_ids_np = np.full(num_agents, -1, dtype=np.int32)
    agent_leisure_ids_np = np.full(num_agents, -1, dtype=np.int32)

    agent_types_np = np.zeros(num_agents, dtype=np.int8)
    agent_home_sigungu_ids_np = np.full(num_agents, -1, dtype=np.int32)
    agent_ages_np = np.zeros(num_agents, dtype=np.int8)

    for agent_elem in agents_root.findall('agent'):
        idx = agent_id_map[agent_elem.get('id')]

        # ID 매핑 보조 함수
        def get_place_index(attr_name):
            place_id_str = agent_elem.get(attr_name)
            if place_id_str:
                return place_id_map.get(place_id_str, -1)
            return -1

        agent_household_ids_np[idx] = get_place_index('household_id')
        agent_workplace_ids_np[idx] = get_place_index('workplace_id')
        agent_preschool_ids_np[idx] = get_place_index('preschool_id')  # 신규
        agent_school_ids_np[idx] = get_place_index('school_id')
        agent_senior_center_ids_np[idx] = get_place_index('senior_center_id')
        # [복원 #15] transport_id 파싱
        agent_transport_ids_np[idx] = get_place_index('transport_id')
        agent_leisure_ids_np[idx] = get_place_index('leisure_id')

        # 에이전트 유형 결정
        agent_type_str = scheduler.get_agent_type(agent_elem)
        agent_types_np[idx] = AGENT_TYPE_MAP.get(agent_type_str, AGENT_TYPE_MAP['unemployed'])

        # 나이 및 시군구 정보 파싱
        try:
            age = int(agent_elem.get('age', 0))
            agent_ages_np[idx] = min(max(age, 0), MAX_AGE - 1)
        except (ValueError, TypeError):
            agent_ages_np[idx] = 0
        try:
            sgg_id = agent_elem.get('sigungu_id')
            if sgg_id:
                agent_home_sigungu_ids_np[idx] = int(sgg_id)
        except ValueError:
            pass

    # === 2. 장소 데이터 파싱 및 이동확대 준비 ===
    place_sigungu_ids_np = np.full(num_places, -1, dtype=np.int32)
    place_types_np = np.full(num_places, INFECTION_PLACE_TYPE_MAP['other'], dtype=np.int8)

    # 이동확대 설정 로드 및 목표 시군구 추출
    travel_config = config.get('travel_params')
    target_sigungu_ids = set()
    if travel_config and travel_config['destinations']:
        target_sigungu_ids = {d['sigungu_id'] for d in travel_config['destinations']}

    popular_destinations_ids = []

    for place_elem in places_root.findall('place'):
        idx = place_id_map[place_elem.get('id')]
        place_type_str = place_elem.get('type')
        sgg_id_str = place_elem.get('sigungu_id')
        sgg_id = -1

        # 시군구 ID 파싱
        if sgg_id_str:
            try:
                sgg_id = int(sgg_id_str)
                place_sigungu_ids_np[idx] = sgg_id
            except ValueError:
                pass

        # 장소 유형 파싱
        if place_type_str:
            place_types_np[idx] = INFECTION_PLACE_TYPE_MAP.get(place_type_str, INFECTION_PLACE_TYPE_MAP['other'])

        # 이동확대 목적지 식별
        # 조건: 목표 시군구에 속하며, 장소 유형이 여가(Leisure) 타입 중 하나일 경우
        if target_sigungu_ids and sgg_id in target_sigungu_ids and place_type_str in LEISURE_TYPES:
            popular_destinations_ids.append(idx)

    # === 3. 데이터 반환 (백엔드 배열로 변환) === (수정)
    data = {
        'num_agents': num_agents,
        'num_places': num_places,
        'agent_id_map': agent_id_map,
        'place_id_map': place_id_map,
        # 속성 배열 백엔드 전환 (NumPy -> CuPy/NumPy)
        'agent_household_id': cp.array(agent_household_ids_np, dtype=cp.int32),
        'agent_workplace_id': cp.array(agent_workplace_ids_np, dtype=cp.int32),
        'agent_preschool_id': cp.array(agent_preschool_ids_np, dtype=cp.int32),  # 신규
        'agent_school_id': cp.array(agent_school_ids_np, dtype=cp.int32),
        'agent_senior_center_id': cp.array(agent_senior_center_ids_np, dtype=cp.int32),
        # [복원 #15] agent_transport_id 복원
        'agent_transport_id': cp.array(agent_transport_ids_np, dtype=cp.int32),
        'agent_leisure_id': cp.array(agent_leisure_ids_np, dtype=cp.int32),

        'agent_types': cp.array(agent_types_np, dtype=cp.int8),
        'agent_home_sigungu_id': cp.array(agent_home_sigungu_ids_np, dtype=cp.int32),
        'place_sigungu_id': cp.array(place_sigungu_ids_np, dtype=cp.int32),
        'place_types': cp.array(place_types_np, dtype=cp.int8),
        'agent_ages': cp.array(agent_ages_np, dtype=cp.int8),
    }

    # 현재 위치 초기화 (집으로 설정)
    data['agent_current_place_id'] = cp.copy(data['agent_household_id'])

    # 이동확대 목적지 배열 추가
    if popular_destinations_ids:
        data['travel_destinations_ids'] = cp.array(popular_destinations_ids, dtype=cp.int32)
        print(f"이동확대 모드 준비 완료: 총 {len(popular_destinations_ids)}개의 인기 이동확대 대상지(장소) 식별됨.")
    else:
        data['travel_destinations_ids'] = None
        if target_sigungu_ids:
            print("참고: 지정된 목표 시군구에서 여가 장소를 찾을 수 없습니다. 이동확대 모드가 활성화되어도 이동확대가 발생하지 않습니다.")

    return data


# ==============================================================================
# (config_loader 나머지 함수들 통합 - 질병 파라미터 로딩은 변경 없음)
# ==============================================================================

# --- 질병 파라미터 로딩 ---

def load_disease_params(file_path, disease_name, cp):
    """질병 파라미터를 로드합니다."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except FileNotFoundError:
        raise FileNotFoundError(f"질병 파라미터 파일 '{file_path}'을 찾을 수 없습니다.")

    disease_node = root.find(f".//Disease[@name='{disease_name}']")
    if disease_node is None:
        # 질병을 찾을 수 없을 경우 예외 발생 (run_simulation에서 처리)
        raise ValueError(f"'{file_path}'에서 '{disease_name}' 질병을 찾을 수 없습니다.")

    params = {}
    # 1. 표준 실수형 파라미터 로드 (TransmissionRate 등)
    for param in disease_node:
        # rate_generator에서 사용하는 Multiplier는 제외
        if param.tag in ['ExposureDuration', 'InfectionDuration', 'SeverityMultiplier', 'FatalityMultiplier']: continue
        if param.text and param.text.strip():
            try:
                params[param.tag] = float(param.text.strip())
            except ValueError:
                pass

    # 2. 기간 파라미터 로드
    params['ExposureDuration'] = _parse_duration_param(disease_node.find('ExposureDuration'), 'ExposureDuration', cp)
    params['InfectionDuration'] = _parse_duration_param(disease_node.find('InfectionDuration'), 'InfectionDuration', cp)

    # [M2 신규] Heterogeneity 파싱
    # <Heterogeneity option="homogeneous|gamma">
    #     <Param name="k_h" value="0.3"/>
    # </Heterogeneity>
    het_node = disease_node.find('Heterogeneity')
    if het_node is not None:
        opt = het_node.get('option', 'homogeneous')
        het_params = {}
        for p_node in het_node.findall('Param'):
            try:
                het_params[p_node.get('name')] = float(p_node.get('value'))
            except (ValueError, TypeError):
                pass
        params['Heterogeneity'] = {'option': opt, 'params': het_params}
    else:
        params['Heterogeneity'] = {'option': 'homogeneous', 'params': {}}

    # [M4 신규] Variants 파싱
    # <Variant name="alpha">
    #     <ExposureDuration>...</ExposureDuration>
    #     <InfectionDuration>...</InfectionDuration>
    #     <Share at_day="0" value="1.0"/>
    #     <Share at_day="120" value="0.0"/>
    # </Variant>
    # [M5 신규] TimescaleHeterogeneity 파싱
    # <TimescaleHeterogeneity k_theta="1.5"/>  → ExposureDuration(weibull)에 적용
    ts_node = disease_node.find('TimescaleHeterogeneity')
    if ts_node is not None:
        try:
            params['TimescaleHeterogeneity'] = {'k_theta': float(ts_node.get('k_theta'))}
            if isinstance(params.get('ExposureDuration'), dict):
                params['ExposureDuration']['apply_timescale'] = True
        except (ValueError, TypeError):
            params['TimescaleHeterogeneity'] = None
    else:
        params['TimescaleHeterogeneity'] = None

    variants = []
    for v_node in disease_node.findall('Variant'):
        v_name = v_node.get('name', f'variant_{len(variants)}')
        v_exp = _parse_duration_param(v_node.find('ExposureDuration'), 'ExposureDuration', cp)
        v_inf = _parse_duration_param(v_node.find('InfectionDuration'), 'InfectionDuration', cp)
        shares = []
        for s_node in v_node.findall('Share'):
            try:
                shares.append((float(s_node.get('at_day')), float(s_node.get('value'))))
            except (ValueError, TypeError):
                pass
        shares.sort(key=lambda t: t[0])
        variants.append({
            'name': v_name, 'ExposureDuration': v_exp,
            'InfectionDuration': v_inf, 'Share': shares,
        })
    params['Variants'] = variants

    # 3. 기본값 설정
    if params['ExposureDuration'] is None:
        print("경고: ExposureDuration 설정이 없어 기본값(5.0일)을 사용합니다.")
        params['ExposureDuration'] = {'type': 'fixed', 'value': 5.0}
    if params['InfectionDuration'] is None:
        print("경고: InfectionDuration 설정이 없어 기본값(14.0일)을 사용합니다.")
        params['InfectionDuration'] = {'type': 'fixed', 'value': 14.0}

    return params


def _parse_duration_param(param_node, param_name, cp):
    """기간 파라미터를 파싱 (고정값 / 이산 분포 / 연속 분포 지원)"""
    if param_node is None:
        return None

    # 1. 확률 분포 구조 확인
    distribution_node = param_node.find('Distribution')
    if distribution_node is not None:
        durations, probabilities = [], []
        for option_node in distribution_node.findall('Option'):
            try:
                dur = float(option_node.get('duration'))
                prob = float(option_node.get('probability'))
                if dur >= 0 and prob > 0:
                    durations.append(dur)
                    probabilities.append(prob)
            except (ValueError, TypeError):
                warnings.warn(f"경고: {param_name}의 잘못된 옵션 발견.")

        if not durations: return None

        # 확률 정규화
        total_prob = sum(probabilities)
        if abs(total_prob - 1.0) > 1e-6 and total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]

        # 백엔드 배열로 변환
        return {
            'type': 'distribution',
            'durations': cp.array(durations, dtype=cp.float32),
            'probabilities': cp.array(probabilities, dtype=cp.float32)
        }

    # [M1 신규] 1.5 연속 분포 구조 확인
    # <Continuous family="lognormal|weibull|gamma|burr12">
    #     <Param name="..." value="..."/> ...
    # </Continuous>
    continuous_node = param_node.find('Continuous')
    if continuous_node is not None:
        family = continuous_node.get('family')
        params = {}
        for p_node in continuous_node.findall('Param'):
            try:
                params[p_node.get('name')] = float(p_node.get('value'))
            except (ValueError, TypeError):
                warnings.warn(f"경고: {param_name}의 잘못된 연속분포 파라미터 발견.")
        return {'type': 'continuous', 'family': family, 'params': params}

    # 2. 고정값 확인
    if param_node.text and param_node.text.strip():
        try:
            value = float(param_node.text.strip())
            return {'type': 'fixed', 'value': value}
        except ValueError:
            return None

    return None


# [M4 신규] 변이 share의 시간 선형 보간
def _interpolate_share(share_points, day):
    """share_points = [(at_day, value), ...] (정렬됨). day 시점의 share 반환."""
    if not share_points:
        return 0.0
    if day <= share_points[0][0]:
        return share_points[0][1]
    if day >= share_points[-1][0]:
        return share_points[-1][1]
    for i in range(len(share_points) - 1):
        d0, v0 = share_points[i]
        d1, v1 = share_points[i + 1]
        if d0 <= day <= d1:
            if d1 == d0:
                return v0
            return v0 + (day - d0) / (d1 - d0) * (v1 - v0)
    return share_points[-1][1]


# --- 정책 파라미터 로딩 ---
# [수정] 마스크 정책(MaskPolicy) 파싱 로직 추가
def load_policy_params(file_path):
    """
    policy.xml 파일을 로드하여 기본 정책, 정책 스케줄, 백신 캠페인 정보를 파싱합니다.
    [수정] 마스크 정책(MaskPolicy) 파싱 기능을 추가합니다.
    """
    # 기본 파라미터 설정
    default_params = {
        'initial_vaccination_coverage': 0.0,
        'vaccine_efficacy_infection': 0.0,
        'vaccine_efficacy_severity': 0.0,
        'diagnosis_delay_days': 999.0,
        'diagnosis_coverage_rate': 0.0,
        'isolation_compliance_rate': 0.0,
        'treatment_coverage_rate': 0.0,
        'treatment_max_trf': 0.0,
        'treatment_time_to_max_effect_days': 1.0,
        # [신규] 마스크 정책 기본값 추가
        'mask_compliance_rate': 0.0, # 마스크 착용률 (Compliance/Adherence)
        'mask_efficacy': 0.0,       # 마스크 효과 (전파 감소율)
    }

    policy_schedule = {}
    vaccine_campaigns = []

    if not file_path or not os.path.exists(file_path):
        warnings.warn(f"정책 파일 '{file_path}'를 찾을 수 없습니다. 기본 정책 설정을 사용합니다.")
        return {'default': default_params, 'schedule': policy_schedule, 'campaigns': vaccine_campaigns}

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print(f"정책 파라미터 파일 로드: {file_path}")

        # 1. 초기 백신 상태 로드 (InitialVaccineStatus)
        initial_node = root.find("InitialVaccineStatus")
        if initial_node is not None:
            mapping = {
                'Coverage': 'initial_vaccination_coverage',
                'EfficacyInfection': 'vaccine_efficacy_infection',
                'EfficacySeverity': 'vaccine_efficacy_severity',
            }
            for tag, param_name in mapping.items():
                node = initial_node.find(tag)
                if node is not None and node.text:
                    try:
                        default_params[param_name] = float(node.text.strip())
                    except ValueError:
                        warnings.warn(f"InitialVaccineStatus의 '{tag}' 값이 유효하지 않습니다: '{node.text}'")

        # 2. 동적 정책 로드 (DiagnosisAndIsolationPolicy, TreatmentPolicy, MaskPolicy)
        # [수정] MaskPolicy 추가
        dynamic_policies = [
            ('DiagnosisAndIsolationPolicy', {
                'DiagnosisDelayDays': 'diagnosis_delay_days',
                'DiagnosisCoverageRate': 'diagnosis_coverage_rate',
                'IsolationComplianceRate': 'isolation_compliance_rate',
            }),
            ('TreatmentPolicy', {
                'TreatmentCoverageRate': 'treatment_coverage_rate',
                'TreatmentMaxTRF': 'treatment_max_trf',
                'TreatmentTimeToMaxEffectDays': 'treatment_time_to_max_effect_days',
            }),
            # [신규] 마스크 정책 로드 설정
            ('MaskPolicy', {
                'MaskComplianceRate': 'mask_compliance_rate',
                'MaskEfficacy': 'mask_efficacy',
            }),
        ]

        for policy_tag, mapping in dynamic_policies:
            for policy_node in root.findall(policy_tag):
                try:
                    start_day_str = policy_node.get('start_day')
                    start_day = int(start_day_str) if start_day_str is not None else 0
                except (ValueError, TypeError):
                    continue

                if start_day not in policy_schedule:
                    policy_schedule[start_day] = {}

                for tag, param_name in mapping.items():
                    node = policy_node.find(tag)
                    if node is not None and node.text:
                        try:
                            value = float(node.text.strip())
                            # 값 범위 확인 (0.0 ~ 1.0, 단 delay/time 등은 제외)
                            if param_name not in ['diagnosis_delay_days', 'treatment_time_to_max_effect_days']:
                                if not (0.0 <= value <= 1.0):
                                     warnings.warn(f"경고: {policy_tag} (Day {start_day})의 '{tag}' 값({value})이 유효 범위(0.0-1.0)를 벗어나 조정됩니다.")
                                value = max(0.0, min(1.0, value))

                            if start_day == 0:
                                default_params[param_name] = value
                            policy_schedule[start_day][param_name] = value
                        except ValueError:
                            pass

        # 3. 백신 캠페인 로드 (VaccineCampaign)
        for campaign_node in root.findall("VaccineCampaign"):
            campaign = {}
            try:
                campaign['start_day'] = int(campaign_node.get('start_day'))
                campaign['duration'] = int(campaign_node.get('duration_days'))
                campaign['target_coverage'] = float(campaign_node.get('target_coverage'))
                if campaign['duration'] <= 0: raise ValueError

                # 선택적 속성: 백신 효과
                efficacy_mapping = {
                    'EfficacyInfection': 'vaccine_efficacy_infection',
                    'EfficacySeverity': 'vaccine_efficacy_severity',
                }
                for tag, param_name in efficacy_mapping.items():
                    node = campaign_node.find(tag)
                    if node is not None and node.text:
                        try:
                            campaign[param_name] = float(node.text.strip())
                        except ValueError:
                            pass
                vaccine_campaigns.append(campaign)

            except (ValueError, TypeError, AttributeError) as e:
                continue


    except (ET.ParseError, FileNotFoundError) as e:
        warnings.warn(f"정책 파일 '{file_path}' 처리 중 오류 발생({e}). 기본 정책 설정을 사용합니다.")
        return {'default': default_params, 'schedule': {}, 'campaigns': []}

    vaccine_campaigns.sort(key=lambda x: x['start_day'])

    return {'default': default_params, 'schedule': policy_schedule, 'campaigns': vaccine_campaigns}


def load_rates_by_age(file_path, rate_type, cp):
    # ... (이 함수는 변경 없음) ...
    """연령별 비율 파일을 범용적으로 로드합니다."""
    if not file_path or not os.path.exists(file_path):
        # 파일이 없어도 경고 후 0으로 초기화하여 진행
        warnings.warn(f"'{rate_type}' 파일 '{file_path}'를 찾을 수 없어 0으로 초기화합니다.")
        return cp.zeros(MAX_AGE, dtype=cp.float32)
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        print(f"시나리오 기반 '{rate_type}' 파일 로드: {file_path}")
    except Exception as e:
        warnings.warn(f"'{rate_type}' 파일 '{file_path}' 파싱 중 오류 발생({e}). 0으로 초기화합니다.")
        return cp.zeros(MAX_AGE, dtype=cp.float32)

    lookup_table_np = np.zeros(MAX_AGE, dtype=np.float32)
    for rate_elem in root.findall('Rate'):
        try:
            age = int(rate_elem.get('age'))
            rate_prob = float(rate_elem.get('value'))
            if 0 <= age < MAX_AGE:
                lookup_table_np[age] = min(max(rate_prob, 0.0), 1.0)
        except (ValueError, TypeError, AttributeError):
            continue
    return cp.array(lookup_table_np)


# --- 초기 감염 설정 ---

def set_initial_infections(sim, initial_infections_file):
    # ... (이 함수는 변경 없음) ...
    """초기 감염자를 설정합니다. (지역 기반 또는 랜덤)"""
    if not initial_infections_file or not os.path.exists(initial_infections_file):
        return _set_initial_infections_random(sim, sim.config.get('initial_infected_count', 10))

    try:
        root = ET.parse(initial_infections_file).getroot()
    except ET.ParseError:
        print(f"경고: 초기 감염 파일 파싱 실패. 랜덤 감염으로 전환합니다.")
        return _set_initial_infections_random(sim, sim.config.get('initial_infected_count', 10))

    regional_targets = {}
    for s in root.findall('sigungu'):
        try:
            sgg_id = int(s.get('id'))
            count = int(s.get('initial'))
            if count > 0:
                regional_targets[sgg_id] = count
        except (ValueError, TypeError):
            continue

    if not regional_targets:
        return _set_initial_infections_random(sim, sim.config.get('initial_infected_count', 10))

    # 성능을 위해 CPU(NumPy)에서 인덱스 계산 (asnumpy 사용)
    cp = sim.cp
    # GPU 데이터일 경우 CPU로 이동
    to_cpu = cp.asnumpy if hasattr(cp, 'asnumpy') else lambda x: x

    sigungu_ids_cpu = to_cpu(sim.agent_home_sigungu_id)
    health_status_cpu = to_cpu(sim.agent_health_status)

    infected_indices_cpu = []
    for sgg_id, target in regional_targets.items():
        candidates = np.where((sigungu_ids_cpu == sgg_id) & (health_status_cpu == STATUS_MAP['S']))[0]
        num_to_infect = min(target, len(candidates))
        if num_to_infect > 0:
            infected_indices_cpu.extend(random.sample(candidates.tolist(), num_to_infect))

    if infected_indices_cpu:
        # 결과를 다시 백엔드 배열로 변환하여 상태 업데이트
        _setup_initial_infected_status(sim, cp.array(infected_indices_cpu, dtype=cp.int32))

    return len(infected_indices_cpu)


def _set_initial_infections_random(sim, num_to_infect):
    # ... (이 함수는 변경 없음) ...
    """랜덤하게 초기 감염자를 설정합니다."""
    cp = sim.cp
    susceptible = cp.where(sim.agent_health_status == STATUS_MAP['S'])[0]
    num_to_infect = min(num_to_infect, len(susceptible))
    if num_to_infect > 0:
        patient_zero_indices = cp.random.choice(susceptible, num_to_infect, replace=False)
        _setup_initial_infected_status(sim, patient_zero_indices)
    return num_to_infect


def _setup_initial_infected_status(sim, infected_indices):
    # ... (이 함수는 변경 없음) ...
    """초기 감염자의 상태를 설정하고 미래 결과를 결정합니다."""
    # disease_model 함수(assign_individual_durations, determine_future_outcome)를 직접 호출합니다.

    cp = sim.cp
    if infected_indices.size == 0: return
    sim.agent_health_status[infected_indices] = STATUS_MAP['I']
    # [M3] 초기 감염자(seed)는 onset_day=0, infector 없음(-1 그대로)
    sim.agent_onset_day[infected_indices] = 0.0

    # 초기 발병 후 경과 시간 (0~1일 랜덤)
    initial_days_passed = cp.random.rand(len(infected_indices)).astype(cp.float32) * 1.0

    target_mask = cp.zeros(sim.num_agents, dtype=cp.bool_)
    target_mask[infected_indices] = True

    # 초기 감염자에게 InfectionDuration 할당
    infection_durations = assign_individual_durations(
        sim, target_mask, sim.disease_params['InfectionDuration']
    )
    sim.agent_infection_duration[infected_indices] = infection_durations

    # 초기 진행 시간이 할당된 기간을 넘지 않도록 보정
    max_onset = cp.maximum(infection_durations - TIME_STEP, TIME_STEP)
    sim.agent_days_since_onset[infected_indices] = cp.minimum(initial_days_passed, max_onset)

    # 미래 결과 결정
    determine_future_outcome(sim, target_mask)


# ==============================================================================
# ==============================================================================
# ===== SECTION 4: Agent Model (agent_model.py 통합 및 수정) ===================
# ==============================================================================
# ==============================================================================

def update_agents_locations(sim):
    """
    스텝별 스케줄, 격리 상태, 이동확대 모드에 따라 에이전트의 위치를 업데이트합니다. (3 스텝 모델로 수정)
    """
    # ... (이 함수는 변경 없음) ...
    cp = sim.cp

    # 현재 스텝의 활동 계획 조회 (sim.step 사용)
    current_activities = sim.schedule_map[sim.agent_types, sim.step]

    # 격리된 에이전트는 활동을 'home'으로 강제
    current_activities = cp.where(sim.agent_is_isolated, PLACE_TYPE_MAP['home'], current_activities)

    # 활동에 따른 목표 장소 설정 (기본값: 현재 위치)
    target_places = sim.agent_current_place_id.copy()

    # --- 1. Home 이동 처리 ---
    is_home_time = (current_activities == PLACE_TYPE_MAP['home'])
    target_places = cp.where(is_home_time, sim.agent_household_id, target_places)

    # --- 1.5. [복원 #15] Transport 이동 처리 (출근/통학) ---
    is_transport_time = (current_activities == PLACE_TYPE_MAP['transport'])
    # transport_id가 -1(미배정)인 경우는 home으로 fallback
    transport_target = cp.where(sim.agent_transport_id >= 0, sim.agent_transport_id, sim.agent_household_id)
    target_places = cp.where(is_transport_time, transport_target, target_places)

    # --- 2. Destination 이동 처리 (직장, 학교, 유치원, 경로당) ---
    is_destination_time = (current_activities == PLACE_TYPE_MAP['destination'])

    if cp.any(is_destination_time):
        # 에이전트 유형별 목적지 매핑 (AGENT_TYPE_MAP의 정수 값 사용)
        destination_map = {
            AGENT_TYPE_MAP['worker']: sim.agent_workplace_id,
            AGENT_TYPE_MAP['preschooler']: sim.agent_preschool_id,  # 신규
            AGENT_TYPE_MAP['student_e']: sim.agent_school_id,
            AGENT_TYPE_MAP['student_mh']: sim.agent_school_id,
            AGENT_TYPE_MAP['senior']: sim.agent_senior_center_id,
        }

        # 각 유형별로 마스크를 생성하여 목적지 할당
        for agent_type_int, id_array in destination_map.items():
            # 현재 스텝이 목적지 시간이고, 해당 유형인 에이전트
            mask = is_destination_time & (sim.agent_types == agent_type_int)
            target_places = cp.where(mask, id_array, target_places)

    # --- 3. Leisure 및 이동확대 모드 처리 (비경제활동인구의 목적지) ---
    is_leisure_time = (current_activities == PLACE_TYPE_MAP['leisure'])
    standard_leisure_targets = sim.agent_leisure_id

    # 이동확대 모드 처리 (기존 로직 유지, Step 1에서만 발생)
    if sim.current_travel_mode and sim.travel_destinations_ids is not None and sim.travel_destinations_ids.size > 0:
        override_prob = sim.current_travel_mode.get('override_probability', 0.0)

        # 여가 시간이고 확률이 0보다 큰 경우에만 처리
        if override_prob > 0 and cp.any(is_leisure_time):
            # 이동확대 대상 에이전트 결정 (확률 기반)
            random_draw = cp.random.rand(sim.num_agents).astype(cp.float32)
            # 여가 시간이고, 확률에 당첨된 경우
            override_mask = is_leisure_time & (random_draw < override_prob)
            num_overrides = int(cp.sum(override_mask))

            if num_overrides > 0:
                # 인기 이동확대 대상지 목록에서 랜덤하게 목적지 샘플링
                num_destinations = len(sim.travel_destinations_ids)
                # num_overrides 수만큼 인덱스를 샘플링하여 목적지 ID를 가져옴
                # cp.random.randint는 [low, high) 범위이므로 high에 num_destinations를 사용
                sampled_indices = cp.random.randint(0, num_destinations, num_overrides)
                sampled_destinations = sim.travel_destinations_ids[sampled_indices]

                # 1. 이동확대 예쩡인 사람 처리 (인덱스를 사용하여 할당)
                override_indices = cp.where(override_mask)[0]
                target_places[override_indices] = sampled_destinations

                # 2. 이동확대 포함되지 않는 나머지 여가 활동자 처리
                standard_leisure_mask = is_leisure_time & (~override_mask)
                target_places = cp.where(standard_leisure_mask, standard_leisure_targets, target_places)

            else:
                # 이동확대 당첨자가 없는 경우 (기본 로직)
                target_places = cp.where(is_leisure_time, standard_leisure_targets, target_places)
        else:
            # 확률이 0이거나 여가 시간이 아닌 경우 (기본 로직)
            target_places = cp.where(is_leisure_time, standard_leisure_targets, target_places)
    else:
        # 이동확대 모드가 비활성화된 경우 또는 목적지가 없는 경우 (기본 로직)
        target_places = cp.where(is_leisure_time, standard_leisure_targets, target_places)

    # 유효한 장소 ID인 경우에만 업데이트 (-1이 아닌 경우)
    valid_targets_mask = (target_places != -1)
    sim.agent_current_place_id = cp.where(valid_targets_mask, target_places, sim.agent_current_place_id)

    # --- 지역간 이동 계산 (기존 로직 유지) ---
    valid_loc = sim.agent_current_place_id >= 0
    max_place_id = sim.num_places - 1

    if max_place_id < 0:  # 장소가 하나도 없는 경우
        return 0

    # 인덱스 범위 초과 방지를 위해 clip 사용
    clipped_place_ids = cp.clip(sim.agent_current_place_id, 0, max_place_id)

    # 현재 위치의 시군구 ID 조회
    if sim.place_sigungu_id.size > 0 and clipped_place_ids.size > 0:
        try:
            current_sigungu = sim.place_sigungu_id[clipped_place_ids]
            # 현재 시군구와 거주지 시군구가 다르고 위치가 유효한 경우 카운트
            inter_regional_movements = cp.sum((current_sigungu != sim.agent_home_sigungu_id) & valid_loc)
        except IndexError:
            # 인덱스 오류 발생 시 (드문 경우) - Step 기반으로 수정
            print(f"경고: Day {sim.day}, Step {sim.step}에 장소 인덱스 오류 발생. 이동량 계산 생략.")
            return 0
    else:
        inter_regional_movements = 0

    # 이동량 반환 (int()로 변환하여 호스트로 이동 및 정수형 변환)
    return int(inter_regional_movements)


# ==============================================================================
# ==============================================================================
# ===== SECTION 5: Disease Model (disease_model.py 통합 및 수정) ===============
# ==============================================================================
# ==============================================================================

# (assign_individual_durations, determine_future_outcome, update_disease_progression 은 변경 없음)

# ==============================================================================
# (disease_model.py 함수들 통합 - 변경 없음 파트)
# ==============================================================================

def assign_individual_durations(sim, mask, duration_param):
    # ... (이 함수는 변경 없음) ...
    """대상 에이전트들에게 고정값 또는 확률 분포에 따라 기간을 할당 (GPU 효율적)"""
    cp = sim.cp
    num_agents = int(cp.sum(mask))
    if num_agents == 0:
        return cp.array([], dtype=cp.float32)

    if duration_param['type'] == 'fixed':
        return cp.full(num_agents, duration_param['value'], dtype=cp.float32)

    elif duration_param['type'] == 'distribution':
        # 역변환 샘플링 (Inverse Transform Sampling)
        durations = duration_param['durations']
        probabilities = duration_param['probabilities']

        # 누적 확률 분포(CDF) 계산
        cumulative_probs = cp.cumsum(probabilities)
        if cumulative_probs.size > 0:
            cumulative_probs[-1] = 1.0  # 부동소수점 오차 보정

        # 0~1 사이의 난수 생성 및 인덱스 탐색
        random_values = cp.random.rand(num_agents).astype(cp.float32)
        indices = cp.searchsorted(cumulative_probs, random_values)
        indices = cp.clip(indices, 0, len(durations) - 1)

        return durations[indices]

    # [M1 신규] 연속 분포 sampling
    elif duration_param['type'] == 'continuous':
        family = duration_param['family']
        p = duration_param['params']
        if family == 'lognormal':
            # X = exp(mu + sigma * Z), Z ~ N(0,1)
            Z = cp.random.standard_normal(num_agents).astype(cp.float32)
            return cp.exp(p['mu'] + p['sigma'] * Z).astype(cp.float32)
        elif family == 'weibull':
            # 역변환: T = scale * (-ln(1-U))^(1/c)
            U = cp.random.rand(num_agents).astype(cp.float32)
            base = (p['scale'] * cp.power(-cp.log(1.0 - U), 1.0 / p['c'])).astype(cp.float32)
            # [M5] 개체별 시간 척도 이질성 (Burr scale-mixture)
            # X|γ ~ Weibull(c, σ·γ^(-1/c)), γ ~ Gamma(k_θ, 1) → X ~ Burr XII(c, k_θ, σ)
            ts = getattr(sim, 'agent_timescale', None)
            if ts is not None and duration_param.get('apply_timescale', False):
                gamma_i = ts[mask]
                base = base * cp.power(gamma_i, -1.0 / p['c'])
            return base.astype(cp.float32)
        elif family == 'gamma':
            # CuPy/NumPy 호환: gamma(shape, scale, size)
            return cp.random.gamma(p['k'], p['scale'], num_agents).astype(cp.float32)
        elif family == 'burr12':
            # 역변환: T = scale * ((1-U)^(-1/k) - 1)^(1/c)
            U = cp.random.rand(num_agents).astype(cp.float32)
            return (p['scale'] * cp.power(cp.power(1.0 - U, -1.0 / p['k']) - 1.0, 1.0 / p['c'])).astype(cp.float32)
        else:
            raise ValueError(f"Unknown continuous family: {family}")


def determine_future_outcome(sim, target_mask):
    # ... (이 함수는 변경 없음) ...
    """감염 시 중증화 및 사망 여부를 결정"""
    cp = sim.cp
    if not cp.any(target_mask): return

    onset_min = sim.disease_params.get('SeverityOnsetMin', DEFAULT_SEVERITY_ONSET_MIN)
    onset_max = sim.disease_params.get('SeverityOnsetMax', DEFAULT_SEVERITY_ONSET_MAX)

    ages = sim.agent_ages[target_mask]

    severity_probs = sim.severity_rate_lookup[ages]  # IHR
    fatality_probs = sim.fatality_rate_lookup[ages]  # IFR

    # 백신 효과 적용
    VE_S = sim.current_policy_params.get('vaccine_efficacy_severity', 0.0)
    if VE_S > 0.0:
        vaccinated_in_target = sim.agent_is_vaccinated[target_mask]
        reduction = cp.where(vaccinated_in_target, 1.0 - VE_S, 1.0).astype(cp.float32)
        severity_probs *= reduction
        fatality_probs *= reduction

    # 중증화 여부 결정
    is_hospitalized_destiny = cp.random.rand(len(ages)).astype(cp.float32) < severity_probs
    num_hospitalized = int(cp.sum(is_hospitalized_destiny))
    if num_hospitalized > 0:
        # 중증화 시점 결정
        onset_days = onset_min + cp.random.rand(num_hospitalized).astype(cp.float32) * (onset_max - onset_min)
        target_indices = cp.where(target_mask)[0]
        hospitalized_indices = target_indices[is_hospitalized_destiny]
        sim.agent_severity_onset_day[hospitalized_indices] = onset_days

    # 사망 여부 결정 (HFR 계산: IFR / IHR)
    hfr = cp.zeros_like(severity_probs)
    valid_mask = severity_probs > 1e-9  # 0으로 나누기 방지
    if cp.any(valid_mask):
        hfr[valid_mask] = fatality_probs[valid_mask] / severity_probs[valid_mask]

    hfr = cp.clip(hfr, 0.0, 1.0)

    # 중증 환자 중 HFR 확률로 사망 결정
    is_fatal_destiny = is_hospitalized_destiny & (cp.random.rand(len(ages)).astype(cp.float32) < hfr)
    num_fatal = int(cp.sum(is_fatal_destiny))
    if num_fatal > 0:
        target_indices = cp.where(target_mask)[0]
        fatal_indices = target_indices[is_fatal_destiny]
        sim.agent_is_fatal[fatal_indices] = True


def update_disease_progression(sim):
    # ... (이 함수는 변경 없음) ...
    """질병 진행 상태를 업데이트합니다. (E->I, I->H, I/H->R/D)"""
    cp = sim.cp

    e_mask = (sim.agent_health_status == STATUS_MAP['E'])
    i_mask = (sim.agent_health_status == STATUS_MAP['I'])
    h_mask = (sim.agent_health_status == STATUS_MAP['H'])

    # 시간 업데이트 (TIME_STEP에 따라 자동으로 1/3일씩 증가)
    sim.agent_days_in_status[e_mask] += TIME_STEP
    onset_mask = (sim.agent_days_since_onset >= 0)
    sim.agent_days_since_onset[onset_mask] += TIME_STEP

    # 1. E -> I 전이
    exposed_finished = e_mask & (sim.agent_exposure_duration >= 0) & \
                       (sim.agent_days_in_status >= sim.agent_exposure_duration)

    if cp.any(exposed_finished):
        sim.agent_health_status[exposed_finished] = STATUS_MAP['I']
        sim.agent_days_in_status[exposed_finished] = 0.0
        sim.agent_days_since_onset[exposed_finished] = TIME_STEP
        sim.agent_exposure_duration[exposed_finished] = -1.0
        # [M3] E->I 전환 시점을 onset_day로 기록
        sim.agent_onset_day[exposed_finished] = sim.day + sim.step * TIME_STEP
        determine_future_outcome(sim, exposed_finished)

    # 2. I -> H 전이
    i_to_h = i_mask & (sim.agent_severity_onset_day > 0) & (sim.agent_days_since_onset >= sim.agent_severity_onset_day)
    if cp.any(i_to_h):
        sim.agent_health_status[i_to_h] = STATUS_MAP['H']

    # 3. I/H -> R/D 전이
    infection_timer_expired = onset_mask & (sim.agent_infection_duration >= 0) & \
                              (sim.agent_days_since_onset >= sim.agent_infection_duration)

    mild_case_finished = i_mask & (sim.agent_severity_onset_day <= 0) & infection_timer_expired
    hospitalized_case_finished = h_mask & infection_timer_expired
    finished_infection = mild_case_finished | hospitalized_case_finished

    newly_r = cp.zeros(sim.num_agents, dtype=cp.bool_)
    newly_d = cp.zeros(sim.num_agents, dtype=cp.bool_)

    if cp.any(finished_infection):
        fatal_mask = finished_infection & sim.agent_is_fatal
        recover_mask = finished_infection & (~sim.agent_is_fatal)

        sim.agent_health_status[fatal_mask] = STATUS_MAP['D']
        sim.agent_health_status[recover_mask] = STATUS_MAP['R']

        # 회복 시 상태 초기화
        sim.agent_days_in_status[recover_mask] = 0.0
        sim.agent_days_since_onset[recover_mask] = -1.0
        sim.agent_severity_onset_day[recover_mask] = -1.0
        sim.agent_exposure_duration[recover_mask] = -1.0
        sim.agent_infection_duration[recover_mask] = -1.0
        sim.agent_is_fatal[recover_mask] = False
        # 정책 상태 초기화
        sim.agent_is_diagnosed[recover_mask] = False
        sim.agent_is_isolated[recover_mask] = False
        sim.agent_is_treated[recover_mask] = False
        sim.agent_treatment_start_time[recover_mask] = -1.0

        newly_r = recover_mask
        newly_d = fatal_mask

    return exposed_finished, i_to_h, newly_r, newly_d


# [수정] 마스크 정책 반영 로직 추가
def run_transmission(sim):
    """
    장소 기반 감염 전파 모델을 실행하고 감염 발생 위치를 추적합니다.
    [수정] 3 스텝 모델 적용 및 마스크 정책 효과를 반영합니다.
    """
    # policy_manager의 함수(calculate_time_dependent_trf)를 직접 호출합니다.

    cp = sim.cp
    # 기본 감염률 (시간당으로 가정)
    base_prob_hourly = sim.disease_params.get('TransmissionRate', 0.0)

    # 감염 위치별 카운트를 저장할 배열 초기화
    num_unique_place_types = len(set(INFECTION_PLACE_TYPE_MAP.values()))
    infection_counts_by_location = cp.zeros(num_unique_place_types, dtype=cp.int32)

    if base_prob_hourly == 0.0:
        return cp.zeros(sim.num_agents, dtype=cp.bool_), infection_counts_by_location

    # 스텝당 감염률로 변환 (1 스텝 = 8시간 가정)
    HOURS_PER_STEP = 6.0  # [복원 #14] 24시간 * TIME_STEP(1/4) = 6시간
    # 1 - (1 - p_h)^8
    prob_per_step = 1.0 - cp.power(1.0 - float(base_prob_hourly), HOURS_PER_STEP)

    # --- [신규] 마스크 정책 효과 적용 (전체 전파율 감소) ---
    mask_compliance = sim.current_policy_params.get('mask_compliance_rate', 0.0)
    mask_efficacy = sim.current_policy_params.get('mask_efficacy', 0.0)

    if mask_compliance > 0 and mask_efficacy > 0:
        # 마스크로 인한 전파 감소 배율 계산: (1 - C*E)^2
        # 이는 마스크가 감염원 제어(Source Control)와 흡입 방어(Inhalation Protection) 양쪽에 효과가 있다는 가정임.
        # 이 배율은 전체 시스템의 평균적인 전파율(β) 감소를 나타냅니다.
        mask_reduction_multiplier = (1.0 - mask_compliance * mask_efficacy) ** 2
        # 기본 전파율(prob_per_step)에 적용 (CPU/GPU 호환성 유지를 위해 명시적 float 변환)
        prob_per_step = float(prob_per_step) * float(mask_reduction_multiplier)
    # --- 마스크 정책 적용 끝 ---


    # 감염원 확인 (I 또는 H 상태)
    infectious_mask = (sim.agent_health_status == STATUS_MAP['I']) | (sim.agent_health_status == STATUS_MAP['H'])
    # [M2] 개체별 전염력 배율 적용
    weights = infectious_mask.astype(cp.float32) * sim.agent_transmission_multiplier

    # 치료 효과 적용 (감염력 감소)
    treated_mask = infectious_mask & sim.agent_is_treated
    if cp.any(treated_mask):
        # [수정] 시간 계산: sim.day + sim.step * TIME_STEP
        current_time = sim.day + sim.step * TIME_STEP
        time_since_trt = current_time - sim.agent_treatment_start_time[treated_mask]
        trf = calculate_time_dependent_trf(sim, cp.maximum(time_since_trt, 0.0))
        weights[treated_mask] *= (1.0 - trf)

    # 활동 중인 감염자 확인
    active_mask = infectious_mask & (sim.agent_current_place_id >= 0)

    if not cp.any(active_mask):
        return cp.zeros(sim.num_agents, dtype=cp.bool_), infection_counts_by_location

    # 장소별 총 감염력 계산 (bincount 사용)
    total_infectiousness = cp.bincount(
        sim.agent_current_place_id[active_mask],
        weights=weights[active_mask],
        minlength=sim.num_places)

    # 에이전트 위치의 감염력 확인
    max_place_id = sim.num_places - 1

    # 장소가 0개인 경우 처리
    if max_place_id < 0:
        return cp.zeros(sim.num_agents, dtype=cp.bool_), infection_counts_by_location

    clipped_place_ids_all = cp.clip(sim.agent_current_place_id, 0, max_place_id)
    infectiousness_at_loc = total_infectiousness[clipped_place_ids_all]

    # 감염 확률 계산 (1 - (1-p_s)^N) (수정: 조정된 prob_per_step 사용)
    # float(prob_per_step)은 이미 마스크 효과가 반영된 값입니다.
    infection_prob = 1.0 - cp.power(1.0 - float(prob_per_step), infectiousness_at_loc.astype(cp.float32))

    # 백신 효과 적용 (감염 예방)
    VE_I = sim.current_policy_params.get('vaccine_efficacy_infection', 0.0)
    if VE_I > 0.0:
        vaccination_reduction = cp.where(sim.agent_is_vaccinated, 1.0 - VE_I, 1.0).astype(cp.float32)
        infection_prob *= vaccination_reduction

    # 신규 노출자 결정
    susceptible_mask = (sim.agent_health_status == STATUS_MAP['S'])
    random_values = cp.random.rand(sim.num_agents).astype(cp.float32)
    newly_exposed_mask = susceptible_mask & (random_values < infection_prob)

    # 개별화된 기간 할당 및 감염 위치 추적
    if cp.any(newly_exposed_mask):

        # 감염 위치 추적 로직
        exposed_location_ids = clipped_place_ids_all[newly_exposed_mask]
        if exposed_location_ids.size > 0 and sim.place_types.size > 0:
            try:
                exposed_location_types = sim.place_types[exposed_location_ids]
                counts = cp.bincount(exposed_location_types, minlength=num_unique_place_types)
                infection_counts_by_location += counts
            except IndexError:
                # [수정] Step 기반 로그
                print(f"경고: Day {sim.day}, Step {sim.step}에 감염 위치 추적 중 장소 인덱스 오류 발생.")

        # [M3] Infector sampling: 각 newly_exposed에 대해 same-place active infector
        # 중 weight(=infectious_mask × multiplier × 치료감소) 비례로 1명 선택
        exposed_idx_dev = cp.where(newly_exposed_mask)[0]
        active_idx_dev  = cp.where(active_mask)[0]
        to_cpu = cp.asnumpy if hasattr(cp, 'asnumpy') else (lambda x: np.asarray(x))
        exposed_idx_cpu   = to_cpu(exposed_idx_dev)
        exposed_place_cpu = to_cpu(sim.agent_current_place_id[exposed_idx_dev])
        active_idx_cpu    = to_cpu(active_idx_dev)
        active_place_cpu  = to_cpu(sim.agent_current_place_id[active_idx_dev])
        active_weight_cpu = to_cpu(weights[active_idx_dev])
        u_cpu = to_cpu(cp.random.rand(exposed_idx_dev.size).astype(cp.float32))

        # 장소별 active 인덱스 그룹화 (CPU)
        actives_at_place = {}
        for k in range(active_idx_cpu.size):
            p = int(active_place_cpu[k])
            actives_at_place.setdefault(p, []).append(k)

        infector_ids_cpu = np.full(exposed_idx_cpu.size, -1, dtype=np.int32)
        for m in range(exposed_idx_cpu.size):
            p = int(exposed_place_cpu[m])
            cands = actives_at_place.get(p)
            if not cands:
                continue
            ws = active_weight_cpu[cands]
            cs = np.cumsum(ws)
            total = float(cs[-1])
            if total <= 0:
                continue
            pick = int(np.searchsorted(cs, u_cpu[m] * total))
            if pick >= len(cands):
                pick = len(cands) - 1
            infector_ids_cpu[m] = active_idx_cpu[cands[pick]]

        sim.agent_infector_id[exposed_idx_dev] = cp.asarray(infector_ids_cpu)
        sim.agent_exposure_day[exposed_idx_dev] = sim.day + sim.step * TIME_STEP

        # 상태 업데이트
        sim.agent_health_status[newly_exposed_mask] = STATUS_MAP['E']
        sim.agent_days_in_status[newly_exposed_mask] = 0.0

        # [M4] Variant-aware duration 할당
        variants = sim.disease_params.get('Variants', [])
        if variants:
            current_day = sim.day + sim.step * TIME_STEP
            shares_np = np.array(
                [_interpolate_share(v['Share'], current_day) for v in variants],
                dtype=np.float32)
            s_sum = float(shares_np.sum())
            if s_sum <= 0:
                shares_np[0] = 1.0; s_sum = 1.0
            shares_np = shares_np / s_sum

            n_exp = int(exposed_idx_dev.size)
            u_v = cp.random.rand(n_exp).astype(cp.float32)
            cum = cp.cumsum(cp.asarray(shares_np, dtype=cp.float32))
            variant_indices = cp.clip(cp.searchsorted(cum, u_v), 0, len(variants) - 1)
            sim.agent_variant_id[exposed_idx_dev] = variant_indices.astype(cp.int8)

            for v_idx, variant in enumerate(variants):
                v_in_exp = (variant_indices == v_idx)
                if not bool(cp.any(v_in_exp)):
                    continue
                v_global_idx = exposed_idx_dev[v_in_exp]
                v_global_mask = cp.zeros(sim.num_agents, dtype=cp.bool_)
                v_global_mask[v_global_idx] = True
                sim.agent_exposure_duration[v_global_idx] = assign_individual_durations(
                    sim, v_global_mask, variant['ExposureDuration'])
                sim.agent_infection_duration[v_global_idx] = assign_individual_durations(
                    sim, v_global_mask, variant['InfectionDuration'])
        else:
            # 기존 단일 분포 경로
            exposure_durations = assign_individual_durations(
                sim, newly_exposed_mask, sim.disease_params['ExposureDuration']
            )
            infection_durations = assign_individual_durations(
                sim, newly_exposed_mask, sim.disease_params['InfectionDuration']
            )
            sim.agent_exposure_duration[newly_exposed_mask] = exposure_durations
            sim.agent_infection_duration[newly_exposed_mask] = infection_durations

    return newly_exposed_mask, infection_counts_by_location


# ==============================================================================
# ==============================================================================
# ===== SECTION 6: Policy Manager (policy_manager.py 통합 및 수정) =============
# ==============================================================================
# ==============================================================================

# (이 섹션은 변경 없음. apply_dynamic_policies가 자동으로 새로운 정책 파라미터를 적용함)

# ==============================================================================
# (policy_manager.py 함수들 통합 - 변경 없음 파트)
# ==============================================================================

def set_initial_vaccination_status(sim):
    # ... (이 함수는 변경 없음) ...
    """초기 백신 접종 상태를 설정합니다."""
    cp = sim.cp
    coverage = sim.current_policy_params.get('initial_vaccination_coverage', 0.0)
    if coverage > 0.0 and sim.num_agents > 0:
        random_values = cp.random.rand(sim.num_agents).astype(cp.float32)
        sim.agent_is_vaccinated = (random_values < coverage)
        return int(cp.sum(sim.agent_is_vaccinated))
    return 0


def apply_dynamic_policies(sim):
    """날짜에 따라 정책 파라미터를 동적으로 업데이트합니다."""
    if sim.day in sim.policy_schedule:
        new_params = sim.policy_schedule[sim.day]
        sim.current_policy_params.update(new_params)
        if sim.day > 0:
            print(f"\n[정책 변경] Day {sim.day}: 새로운 정책이 적용되었습니다: {new_params}")


def apply_vaccination_campaign(sim):
    # ... (이 함수는 변경 없음) ...
    """백신 캠페인을 적용하여 일일 접종을 수행합니다."""
    cp = sim.cp
    daily_vaccinations = 0

    if sim.num_agents == 0: return

    for campaign in sim.vaccine_campaigns:
        start_day = campaign.get('start_day')
        duration = campaign.get('duration')
        target_coverage = campaign.get('target_coverage')

        if start_day is None or duration is None or target_coverage is None: continue

        end_day = start_day + duration

        if start_day <= sim.day < end_day:
            # 현재 접종률 계산
            current_coverage = float(cp.sum(sim.agent_is_vaccinated) / sim.num_agents)

            if current_coverage < target_coverage:
                remaining_days = end_day - sim.day
                if remaining_days > 0:
                    # 남은 양을 남은 기간으로 균등 분배 (매일 재계산)
                    daily_coverage_target = (target_coverage - current_coverage) / remaining_days
                    daily_target_count = int(round(daily_coverage_target * sim.num_agents))

                    unvaccinated_mask = ~sim.agent_is_vaccinated
                    num_to_vaccinate = min(daily_target_count, int(cp.sum(unvaccinated_mask)))

                    if num_to_vaccinate > 0:
                        # GPU/CPU에서 직접 샘플링
                        unvaccinated_indices = cp.where(unvaccinated_mask)[0]
                        vaccinated_indices = cp.random.choice(unvaccinated_indices, num_to_vaccinate, replace=False)
                        sim.agent_is_vaccinated[vaccinated_indices] = True
                        daily_vaccinations += num_to_vaccinate

                        # 캠페인 정의된 백신 효과 업데이트 (옵션)
                        if 'vaccine_efficacy_infection' in campaign:
                            sim.current_policy_params['vaccine_efficacy_infection'] = campaign['vaccine_efficacy_infection']
                        if 'vaccine_efficacy_severity' in campaign:
                            sim.current_policy_params['vaccine_efficacy_severity'] = campaign['vaccine_efficacy_severity']

    # Day 0 이후에만 로그 출력
    if daily_vaccinations > 0 and sim.day > 0:
        new_coverage = float(cp.sum(sim.agent_is_vaccinated) / sim.num_agents)
        print(f"\n[백신 캠페인] Day {sim.day}: {daily_vaccinations}명 추가 접종. (총 접종률: {new_coverage * 100:.2f}%)")


def apply_health_policies(sim):
    """
    진단, 격리, 치료 정책을 적용합니다.
    [수정] 3 스텝 모델에 맞게 확률 계산 및 시간 기록을 수정합니다.
    (마스크 정책은 개별 적용이 아니므로 이 함수에서는 처리하지 않고 run_transmission에서 처리함)
    """
    cp = sim.cp
    delay = sim.current_policy_params.get('diagnosis_delay_days', 999.0)
    coverage = sim.current_policy_params.get('diagnosis_coverage_rate', 0.0)

    if coverage <= 0: return

    # 진단 대상자 선정
    candidates = ((sim.agent_health_status == STATUS_MAP['I']) | (sim.agent_health_status == STATUS_MAP['H'])) & \
                 (~sim.agent_is_diagnosed) & (sim.agent_days_since_onset >= delay)

    if cp.any(candidates):
        # [수정] 스텝당 진단 확률 계산 (1일 = 3스텝)
        try:
            # coverage가 1에 매우 가까우면 계산 오류 발생 가능성 있어 0.9999로 제한
            # 1/24.0 대신 TIME_STEP(1/3.0) 사용. P_step = 1 - (1-P_day)^(1/3)
            prob_per_step = 1.0 - (1.0 - min(coverage, 0.9999)) ** TIME_STEP
        except (ValueError, ZeroDivisionError):
            prob_per_step = 0.0

        random_draw = cp.random.rand(sim.num_agents).astype(cp.float32)
        newly_diagnosed = candidates & (random_draw < prob_per_step)

        if cp.any(newly_diagnosed):
            sim.agent_is_diagnosed[newly_diagnosed] = True

            # 격리 적용
            iso_compliance = sim.current_policy_params.get('isolation_compliance_rate', 0.0)
            if iso_compliance > 0:
                random_draw_iso = cp.random.rand(sim.num_agents).astype(cp.float32)
                start_isolation = newly_diagnosed & (random_draw_iso < iso_compliance)
                sim.agent_is_isolated[start_isolation] = True

            # 치료 적용
            trt_coverage = sim.current_policy_params.get('treatment_coverage_rate', 0.0)
            if trt_coverage > 0:
                random_draw_trt = cp.random.rand(sim.num_agents).astype(cp.float32)
                start_treatment = newly_diagnosed & (random_draw_trt < trt_coverage)
                if cp.any(start_treatment):
                    sim.agent_is_treated[start_treatment] = True
                    # [수정] 치료 시작 시점 기록 (시간 계산 수정)
                    current_time = sim.day + sim.step * TIME_STEP
                    sim.agent_treatment_start_time[start_treatment] = current_time


def calculate_time_dependent_trf(sim, time_since_treatment):
    # ... (이 함수는 변경 없음) ...
    """치료 경과 시간에 따른 감염력 감소 효과(TRF)를 계산합니다."""
    cp = sim.cp
    max_trf = sim.current_policy_params.get('treatment_max_trf', 0.0)
    time_to_max = sim.current_policy_params.get('treatment_time_to_max_effect_days', 1.0)

    if time_to_max <= 0:
        # 0으로 나누기 방지.
        return cp.zeros_like(time_since_treatment)

    # 선형 증가 모델
    trf = (time_since_treatment / time_to_max) * max_trf
    return cp.clip(trf, 0.0, max_trf)


# ==============================================================================
# ==============================================================================
# ===== SECTION 7: Reporting (reporting.py 통합 및 수정) =======================
# ==============================================================================
# ==============================================================================

# (이 섹션은 변경 없음)

def initialize_log_files(config):
    # ... (이 함수는 변경 없음) ...
    """로그 파일 헤더를 초기화합니다. (수정: hour 대신 step 사용)"""
    output_dir = config['output_dir']
    # 스텝별 로그 파일
    log_path = os.path.join(output_dir, config['log_file'])

    # [수정] 기본 헤더 정의: hour 대신 step 사용
    header = ['day', 'step', 'S', 'E', 'I', 'H', 'R', 'D', 'New_E', 'New_I', 'New_H', 'New_R', 'New_D',
              'Total_Vaccinated', 'Total_Diagnosed', 'Current_Isolated', 'Current_Treated', 'Travel_Mode_Active']

    # 감염 위치별 헤더 추가
    sorted_infection_headers = [v for k, v in sorted(INFECTION_PLACE_HEADERS.items())]
    header.extend(sorted_infection_headers)

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # 일별 지역 통계 로그 파일
    regional_path = os.path.join(output_dir, 'daily_regional_cases.csv')
    with open(regional_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['day', 'sigungu_id', 'S', 'E', 'I', 'H', 'R', 'D', 'new_cases', 'new_hospitalized'])

    # 일별 이동량 로그 파일
    movement_path = os.path.join(output_dir, 'daily_movement_log.csv')
    with open(movement_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['day', 'total_inter_regional_movements'])


def save_population_data(sim):
    # ... (이 함수는 변경 없음) ...
    """지역별 인구 데이터를 XML 파일로 저장합니다."""
    cp = sim.cp
    output_path = os.path.join(sim.config['output_dir'], 'sigungu_population.xml')

    # GPU 데이터를 CPU로 이동
    to_cpu = cp.asnumpy if hasattr(cp, 'asnumpy') else lambda x: x
    sigungu_ids_cpu = to_cpu(sim.agent_home_sigungu_id)

    # 유효한 ID만 필터링하여 인구수 집계
    valid_ids = sigungu_ids_cpu[sigungu_ids_cpu >= 0]
    if valid_ids.size > 0:
        unique_ids, counts = np.unique(valid_ids, return_counts=True)
    else:
        unique_ids, counts = [], []

    # XML 생성 및 저장
    root = ET.Element('populations')
    for sid, pop in zip(unique_ids, counts):
        ET.SubElement(root, 'sigungu', {'id': str(sid), 'population': str(pop)})

    xml_str = ET.tostring(root, encoding='utf-8')
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml_str = parsed_xml.toprettyxml(indent="  ", encoding="utf-8")

    with open(output_path, "wb") as f:
        f.write(pretty_xml_str)


def log_step_stats(sim, new_e, new_i, new_h, new_r, new_d, infection_counts_by_loc):
    # ... (이 함수는 변경 없음) ...
    """
    스텝별 통계를 기록합니다.
    [수정] hour 대신 step을 기록합니다.
    """
    cp = sim.cp
    status_counts = cp.bincount(sim.agent_health_status, minlength=len(STATUS_MAP))

    # [수정] day와 step 기록
    log_data = [sim.day, sim.step]

    # GPU/CPU 모드에 따라 데이터 가져오기
    if sim.IS_GPU_MODE and hasattr(status_counts, 'get'):
        counts = status_counts.get()
        log_data.extend([int(count) for count in counts])
    else:
        log_data.extend([int(count) for count in status_counts])

    # 발생 건수 추가
    log_data.extend([int(cp.sum(m)) for m in [new_e, new_i, new_h, new_r, new_d]])

    # 정책 상태 추가
    log_data.extend([
        int(cp.sum(sim.agent_is_vaccinated)),
        int(cp.sum(sim.agent_is_diagnosed)),
        int(cp.sum(sim.agent_is_isolated)),
        int(cp.sum(sim.agent_is_treated))
    ])

    # 이동확대 모드 활성화 상태 추가 (1 또는 0)
    log_data.append(1 if sim.current_travel_mode else 0)

    # 감염 위치별 통계 추가
    if sim.IS_GPU_MODE and hasattr(infection_counts_by_loc, 'get'):
        location_counts = infection_counts_by_loc.get()
    else:
        location_counts = infection_counts_by_loc

    num_unique_place_types = len(INFECTION_PLACE_HEADERS)
    for i in range(num_unique_place_types):
        try:
            log_data.append(int(location_counts[i]))
        except IndexError:
            log_data.append(0)

    log_path = os.path.join(sim.config['output_dir'], sim.config['log_file'])
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(log_data)


def log_daily_regional_cases(sim, daily_newly_confirmed_mask, daily_newly_hospitalized_mask):
    # ... (이 함수는 변경 없음) ...
    """일별 지역 통계를 기록합니다. (Pandas 사용)"""
    cp = sim.cp

    # GPU 데이터를 CPU로 이동
    to_cpu = cp.asnumpy if hasattr(cp, 'asnumpy') else lambda x: x

    sigungu_ids_cpu = to_cpu(sim.agent_home_sigungu_id)
    health_status_cpu = to_cpu(sim.agent_health_status)
    newly_confirmed_mask_cpu = to_cpu(daily_newly_confirmed_mask)
    newly_hospitalized_mask_cpu = to_cpu(daily_newly_hospitalized_mask)

    valid_mask = sigungu_ids_cpu >= 0
    df = pd.DataFrame({
        'sigungu_id': sigungu_ids_cpu[valid_mask],
        'status': health_status_cpu[valid_mask],
        'new_cases': newly_confirmed_mask_cpu[valid_mask],
        'new_hospitalized': newly_hospitalized_mask_cpu[valid_mask]
    })

    # 상태별 인구수 및 신규 사례 집계
    status_counts = pd.crosstab(df['sigungu_id'], df['status']).rename(columns={v: k for k, v in STATUS_MAP.items()})
    regional_summary = df.groupby('sigungu_id')[['new_cases', 'new_hospitalized']].sum()

    # 결과 병합 및 정리
    result_df = pd.concat([status_counts, regional_summary], axis=1).fillna(0)
    for col in list(STATUS_MAP.keys()) + ['new_cases', 'new_hospitalized']:
        if col not in result_df: result_df[col] = 0
        result_df[col] = result_df[col].astype(int)

    log_path = os.path.join(sim.config['output_dir'], 'daily_regional_cases.csv')
    result_df = result_df.reset_index()
    result_df['day'] = sim.day
    result_df.to_csv(log_path, mode='a', header=False, index=False,
                     columns=['day', 'sigungu_id', 'S', 'E', 'I', 'H', 'R', 'D', 'new_cases', 'new_hospitalized'])


def log_daily_movements(sim, total_movements):
    # ... (이 함수는 변경 없음) ...
    """일별 이동량을 기록합니다."""
    log_path = os.path.join(sim.config['output_dir'], 'daily_movement_log.csv')
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([sim.day, total_movements])


def save_gen_interval(sim):
    """[M3] 시뮬레이션 종료 후 (parent, child, T) CSV 출력.
    T_days = child_onset - parent_onset.
    """
    cp = sim.cp
    to_cpu = cp.asnumpy if hasattr(cp, 'asnumpy') else (lambda x: np.asarray(x))

    pairs_mask = sim.agent_infector_id >= 0
    if not bool(cp.any(pairs_mask)):
        print("[M3] 전염 페어 없음. sim_gen_interval.csv 생성 안 함.")
        return

    child_ids_dev = cp.where(pairs_mask)[0]
    parent_ids_dev = sim.agent_infector_id[child_ids_dev]
    child_onset_dev = sim.agent_onset_day[child_ids_dev]
    parent_onset_dev = sim.agent_onset_day[parent_ids_dev]

    df = pd.DataFrame({
        'child_id':       to_cpu(child_ids_dev),
        'parent_id':      to_cpu(parent_ids_dev),
        'child_onset':    to_cpu(child_onset_dev),
        'parent_onset':   to_cpu(parent_onset_dev),
        # [M4] 변이 ID (Variants 미정의 시 모두 0)
        'child_variant':  to_cpu(sim.agent_variant_id[child_ids_dev]),
        'parent_variant': to_cpu(sim.agent_variant_id[parent_ids_dev]),
        # [FL-EXT] 연합학습 확장: 시군구별 분할·연령 분석용 컬럼 추가
        'child_sigungu':  to_cpu(sim.agent_home_sigungu_id[child_ids_dev]),
        'parent_sigungu': to_cpu(sim.agent_home_sigungu_id[parent_ids_dev]),
        'child_age':      to_cpu(sim.agent_ages[child_ids_dev]),
        'child_exposure_day': to_cpu(sim.agent_exposure_day[child_ids_dev]),
    })
    # 부모가 아직 onset 전(즉 onset_day == -1)인 케이스 분리: T_days는 미정의
    valid = (df['parent_onset'] >= 0) & (df['child_onset'] >= 0)
    df['T_days'] = df['child_onset'] - df['parent_onset']
    df.loc[~valid, 'T_days'] = float('nan')

    out_path = os.path.join(sim.config['output_dir'], 'sim_gen_interval.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[M3] 전염 페어 {len(df)}건 저장 (유효 T {int(valid.sum())}건): {out_path}")


def save_daily_seihr_graph(log_file_path, output_png_path):
    # ... (이 함수는 변경 없음) ...
    """
    로그 파일을 읽어 일별 SEIHR 그래프를 생성하고 저장합니다.
    [수정] 2025-11-11: 사용자 이미지 요구사항 반영 (배경 제거, 라인/레이블/범례 수정, 축 정의 추가)
    [수정] 2025-11-11: 한글 폰트 깨짐 문제 수정
    [수정] 2025-11-11: 폰트 크기 2배 및 범례/텍스트 정렬 수정
    """
    if plt is None:
        print("오류: matplotlib가 설치되지 않아 그래프를 생성할 수 없습니다.")
        return

    # --- [신규] 한글 폰트 설정 ---
    try:
        system_name = platform.system()
        if system_name == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif system_name == 'Darwin': # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        else: # Linux 등
            # 'NanumGothic'이 설치되어 있다고 가정하거나,
            # 'sans-serif'에 한글 폰트가 포함되도록 시스템 설정이 필요할 수 있음
            # 'Malgun Gothic'을 기본으로 설정 (Windows 환경을 가정)
            plt.rcParams['font.family'] = 'Malgun Gothic'

        # 폰트 설정 후 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"경고: 한글 폰트 설정 중 오류 발생: {e}. 폰트가 깨질 수 있습니다.")
    # --- 폰트 설정 끝 ---

    try:
        print(f"\n'{log_file_path}'에서 데이터를 읽어 그래프를 생성합니다...")
        df = pd.read_csv(log_file_path)
        if df.empty or df['day'].nunique() == 0:
            print("경고: 로그 파일이 비어 있어 그래프를 생성할 수 없습니다.")
            return

        # 1. 데이터 전처리
        # [수정] hour 대신 step을 기준으로 하루의 마지막 상태를 가져옴
        daily_df = df.loc[df.groupby('day')['step'].idxmax()]

        # 2. 그래프 설정
        # [수정] ggplot 스타일 제거, 흰색 배경 설정
        # plt.style.use('ggplot') # 제거
        fig, ax1 = plt.subplots(figsize=(28, 12))
        fig.patch.set_facecolor('white')
        ax1.set_facecolor('white')

        colors = {'S': '#3498db', 'E': '#f39c12', 'I': '#e74c3c', 'H': '#9b59b6', 'R': '#2ecc71', 'D': '#34495e'}

        # 3. SIR 상태 플롯 (왼쪽 Y축) - [수정] S, I, R만 남기고 레이블 변경
        ax1.plot(daily_df['day'], daily_df['S'], label='감염 대상군', color=colors['S'], linewidth=2)
        ax1.plot(daily_df['day'], daily_df['I'], label='감염군', color=colors['I'], linewidth=2)
        ax1.plot(daily_df['day'], daily_df['R'], label='회복군', color=colors['R'], linewidth=2)

        # [수정] E, H, D, Total_Vaccinated, Current_Isolated, Travel_Mode_Active 플롯 제거
        # ... (제거된 플롯) ...

        # [수정] 왼쪽 Y축 및 X축 레이블 변경, 틱 방향 설정, 폰트 크기 2배
        ax1.set_xlabel('T (day)', fontsize=32) # 16 -> 32
        ax1.set_ylabel('P (persons)', fontsize=32) # 16 -> 32
        ax1.tick_params(axis='y', labelsize=28, direction='in') # 14 -> 28
        ax1.tick_params(axis='x', labelsize=28, direction='in') # 14 -> 28
        ax1.grid(False) # [수정] 그리드 제거

        # 4. [수정] 신규 사례 막대 그래프 (오른쪽 Y축) 제거
        # ...

        # Y축 범위 설정
        ax1.set_ylim(bottom=0)

        # 5. 제목 및 범례 설정
        ax1.set_title('Daily SIR Progression (Simplified 3-Step Model)', fontsize=44, pad=20) # [수정] 제목 단순화, 22 -> 44

        # [수정] 범례 통합 및 위치 조정 (ax2 제거됨), 폰트 크기 2배
        lines, labels = ax1.get_legend_handles_labels()
        # [수정] 범례 위치를 그래프 외부로 조정 (Y위치를 0.7로 살짝 올림, loc='upper left'로 변경)
        ax1.legend(lines, labels, loc='upper left', fontsize=24, bbox_to_anchor=(1.05, 0.7)) # 12 -> 24, Y: 0.5 -> 0.6

        # [신규] P와 T 정의를 그래프 여백(우측 하단)에 추가, 폰트 크기 2배, X위치 조정
        # [수정] fig.text를 ax1.text로 변경하여 축(ax1) 기준으로 정렬 (va='top' 사용)
        ax1.text(1.05, 0.5, "P: Population", transform=ax1.transAxes, fontsize=24, ha='left', va='top')
        ax1.text(1.05, 0.45, "T: Time", transform=ax1.transAxes, fontsize=24, ha='left', va='top')


        # X축 설정
        if len(daily_df['day']) > 0:
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # 6. 레이아웃 조정 및 저장
        # [수정] 범례 및 P/T 텍스트 공간 확보를 위해 rect 조정 (0.85 유지)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(output_png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"성공: 일별 요약 그래프를 '{output_png_path}'에 저장했습니다.")
        plt.close(fig)

    except Exception as e:
        print(f"오류: 그래프 생성 중 예기치 않은 오류 발생: {e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# ==============================================================================
# ===== SECTION 8: Simulation Core (simulation_core.py 통합 및 수정) ============
# ==============================================================================
# ==============================================================================

class Simulation:
    # ... (이 클래스는 변경 없음) ...
    def __init__(self, config, backend_module, is_gpu_mode):
        self.config = config
        self.day = 0
        self.step = 0  # [수정] hour 대신 step 사용
        self.cp = backend_module  # NumPy or CuPy
        self.IS_GPU_MODE = is_gpu_mode
        self.nvml_handle = None

        print("시뮬레이션 초기화를 시작합니다 (모듈화된 SEIHRD 모델)...")
        os.makedirs(config['output_dir'], exist_ok=True)

        # 1. 스케줄러 로드 (단순화 모델 로드)
        self.scheduler = load_scheduler(config['schedules_file'])

        # 1.1 이동확대 설정 로드 (run_simulation에서 로드되어 config로 전달됨)
        self.travel_data = config.get('travel_params', {'schedule': [], 'destinations': []})
        self.travel_schedule = self.travel_data['schedule']
        self.current_travel_mode = None  # 현재 활성화된 이동확대 모드 정보

        # 2. 기본 데이터 로드 및 배열 초기화 (이동확대 목적지 정보도 함께 로드됨)
        data = load_and_prepare_data(config, self.scheduler, self.cp)
        self._initialize_simulation_arrays(data)
        # 3 스텝 스케줄 맵 준비
        self.schedule_map = prepare_schedule_map(self.scheduler, self.cp)

        # 3. 질병 파라미터 및 중증/치명률 로드
        self.disease_params = load_disease_params(
            config['disease_params_file'], config['disease_name'], self.cp
        )
        # [M2/M3/M4] disease_params 로드 후 M-series 배열 초기화
        self._init_m_series_arrays()
        self.severity_rate_lookup = load_rates_by_age(
            config.get('severity_rates_file'), '중증화율(severity)', self.cp
        )
        self.fatality_rate_lookup = load_rates_by_age(
            config.get('fatality_rates_file'), '치명률(fatality)', self.cp
        )

        # 4. 정책 파라미터 로드 및 초기화
        self.policy_data = config.get('policy_params', {'default': {}, 'schedule': {}, 'campaigns': []})
        self.current_policy_params = self.policy_data['default'].copy()
        self.policy_schedule = self.policy_data['schedule']
        self.vaccine_campaigns = self.policy_data['campaigns']

        # 5. 로깅 및 모니터링 초기화
        initialize_log_files(config)
        save_population_data(self)
        self._initialize_gpu_monitoring()

        # 6. 초기 상태 설정 (Day 0)
        set_initial_vaccination_status(self)
        apply_dynamic_policies(self)
        apply_vaccination_campaign(self)
        self._update_travel_mode()

        # 초기 감염 설정
        initial_infected_count = set_initial_infections(
            self, config.get('initial_infections_file')
        )

        # 초기화 완료 로그
        self._log_initialization_summary(initial_infected_count)

    def _initialize_simulation_arrays(self, data):
        """Simulation 클래스에서 사용할 배열들을 초기화합니다. (수정)"""
        self.num_agents = data['num_agents']
        self.num_places = data['num_places']
        self.agent_id_map = data['agent_id_map']
        self.place_id_map = data['place_id_map']

        # 에이전트 속성 배열 할당 (수정: preschool_id 추가, transport_id 제거)
        self.agent_household_id = data['agent_household_id']
        self.agent_current_place_id = data['agent_current_place_id']
        self.agent_workplace_id = data['agent_workplace_id']
        self.agent_preschool_id = data['agent_preschool_id']  # 신규
        self.agent_school_id = data['agent_school_id']
        self.agent_senior_center_id = data['agent_senior_center_id']
        # [복원 #15] self.agent_transport_id 복원
        self.agent_transport_id = data['agent_transport_id']
        self.agent_leisure_id = data['agent_leisure_id']

        self.agent_types = data['agent_types']
        self.agent_home_sigungu_id = data['agent_home_sigungu_id']
        self.place_sigungu_id = data['place_sigungu_id']
        self.place_types = data['place_types']
        self.agent_ages = data['agent_ages']

        # 이동확대 대상지 배열 할당
        self.travel_destinations_ids = data['travel_destinations_ids']

        # 시뮬레이션 상태 배열 초기화
        self.agent_health_status = self.cp.zeros(self.num_agents, dtype=self.cp.int8)
        self.agent_days_in_status = self.cp.zeros(self.num_agents, dtype=self.cp.float32)
        self.agent_days_since_onset = self.cp.full(self.num_agents, -1.0, dtype=self.cp.float32)
        self.agent_severity_onset_day = self.cp.full(self.num_agents, -1.0, dtype=self.cp.float32)
        self.agent_exposure_duration = self.cp.full(self.num_agents, -1.0, dtype=self.cp.float32)
        self.agent_infection_duration = self.cp.full(self.num_agents, -1.0, dtype=self.cp.float32)

        # 에이전트 상태 플래그
        self.agent_is_fatal = self.cp.zeros(self.num_agents, dtype=self.cp.bool_)
        self.agent_is_vaccinated = self.cp.zeros(self.num_agents, dtype=self.cp.bool_)
        self.agent_is_diagnosed = self.cp.zeros(self.num_agents, dtype=self.cp.bool_)
        self.agent_is_isolated = self.cp.zeros(self.num_agents, dtype=self.cp.bool_)
        self.agent_is_treated = self.cp.zeros(self.num_agents, dtype=self.cp.bool_)
        self.agent_treatment_start_time = self.cp.full(self.num_agents, -1.0, dtype=self.cp.float32)

        # [M2/M3/M4 신규 init] disease_params 로드 후 호출되는 _init_m_series_arrays에서 처리

    def _init_m_series_arrays(self):
        """[M2/M3/M4] disease_params 로드 후 호출. self.num_agents와 self.disease_params 필요."""
        # [M2] 개체별 전염력 배율 (Heterogeneity)
        het = self.disease_params.get('Heterogeneity', {'option': 'homogeneous', 'params': {}})
        if het['option'] == 'gamma':
            k_h = het['params'].get('k_h', 1.0)
            # mean=1 정규화: Gamma(k_h, 1/k_h) -> 평균 1, 분산 1/k_h
            self.agent_transmission_multiplier = self.cp.random.gamma(
                k_h, 1.0 / k_h, self.num_agents).astype(self.cp.float32)
        elif het['option'] == 'gamma_unscaled':
            # [M2 후속/Option (b)] unscaled: Gamma(k_h, 1) -> 평균 = 분산 = k_h (R_0 함께 변함)
            k_h = het['params'].get('k_h', 1.0)
            self.agent_transmission_multiplier = self.cp.random.gamma(
                k_h, 1.0, self.num_agents).astype(self.cp.float32)
        elif het['option'] == 'lognormal':
            # [M2 후속/B2] mean=1 정규화: w_i = exp(-sigma^2/2 + sigma*Z), Z~N(0,1) -> 평균 1
            sigma = het['params'].get('sigma', 1.0)
            Z = self.cp.random.standard_normal(self.num_agents).astype(self.cp.float32)
            self.agent_transmission_multiplier = self.cp.exp(-0.5*sigma*sigma + sigma*Z).astype(self.cp.float32)
        elif het['option'] == 'pareto':
            # [M2 후속/B3] mean=1 정규화: alpha>1 필요. w_i = (alpha-1) * Pareto(alpha)
            alpha = het['params'].get('alpha', 2.0)
            if alpha <= 1.0:
                raise ValueError(f"Pareto: alpha={alpha} <= 1 -> 평균 미정의")
            self.agent_transmission_multiplier = ((alpha - 1.0) *
                self.cp.random.pareto(alpha, self.num_agents)).astype(self.cp.float32)
        else:  # 'homogeneous' (default) — 모두 1.0
            self.agent_transmission_multiplier = self.cp.ones(self.num_agents, dtype=self.cp.float32)

        # [M5 신규] 개체별 시간 척도 이질성 (Burr scale-mixture의 Θ)
        # γ_i ~ Gamma(k_θ, 1). ExposureDuration이 weibull이고 apply_timescale이면
        # 개별 척도 = σ·γ_i^(-1/c) → 주변분포가 정확히 Burr XII(c, k_θ, σ)
        ts_cfg = self.disease_params.get('TimescaleHeterogeneity')
        if ts_cfg and ts_cfg.get('k_theta', 0) > 0:
            k_theta = ts_cfg['k_theta']
            self.agent_timescale = self.cp.random.gamma(
                k_theta, 1.0, self.num_agents).astype(self.cp.float32)
            print(f'[M5] 개체별 시간척도 이질성 활성: γ ~ Gamma(k_θ={k_theta}, 1), '
                  f'평균={float(self.agent_timescale.mean()):.3f}')
        else:
            self.agent_timescale = None

        # [M3] 전염 페어 로깅: infector_id, exposure_day, onset_day
        # generation interval T = onset_day[child] - onset_day[parent]
        self.agent_infector_id = self.cp.full(self.num_agents, -1, dtype=self.cp.int32)
        self.agent_exposure_day = self.cp.full(self.num_agents, -1.0, dtype=self.cp.float32)
        self.agent_onset_day = self.cp.full(self.num_agents, -1.0, dtype=self.cp.float32)

        # [M4] 변이 ID (default = 0; Variants 미정의 시 모두 0 유지)
        self.agent_variant_id = self.cp.zeros(self.num_agents, dtype=self.cp.int8)

    # 이동확대 모드 업데이트 함수 (변경 없음)
    def _update_travel_mode(self):
        """현재 날짜(self.day)를 기준으로 이동확대 모드를 확인하고 업데이트합니다."""
        active_mode = None
        for mode in self.travel_schedule:
            # end_day는 포함됨
            if mode['start_day'] <= self.day <= mode['end_day']:
                active_mode = mode
                break  # 먼저 발견된 활성 모드를 적용 (겹치는 경우 시작일이 빠른 것 우선)

        # 상태가 변경되었을 때만 로그 출력 및 업데이트
        if active_mode != self.current_travel_mode:
            # 로그 출력 (상태 변경 시점)
            if active_mode:
                prob = active_mode['override_probability'] * 100
                # Day 0이거나 Day > 0일 때 출력
                print(f"\n[이동확대 모드 활성화] Day {self.day}: '{active_mode['name']}' 모드 적용 (이동확대 확률: {prob:.1f}%)")
            elif self.day > 0 and self.current_travel_mode is not None:
                # Day 0이 아니고, 이전 상태가 활성이었고 현재 비활성인 경우
                print(f"\n[이동확대 모드 변경] Day {self.day}: 이동확대 모드 종료. 일반 스케줄로 복귀.")

            self.current_travel_mode = active_mode

    # --- GPU 모니터링 관련 함수들 (원본과 동일) ---
    def _initialize_gpu_monitoring(self):
        """GPU 모니터링을 초기화합니다. (pynvml 사용)"""
        if self.IS_GPU_MODE:
            try:
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
                nvmlInit()
                self.nvml_handle = nvmlDeviceGetHandleByIndex(0)
                self.max_gpu_memory_used = 0
            except Exception:
                self.nvml_handle = None

    def _update_gpu_memory_log(self):
        """GPU 메모리 사용량을 업데이트합니다."""
        if self.nvml_handle:
            try:
                from pynvml import nvmlDeviceGetMemoryInfo
                info = nvmlDeviceGetMemoryInfo(self.nvml_handle)
                self.max_gpu_memory_used = max(self.max_gpu_memory_used, info.used)
            except Exception:
                self.nvml_handle = None

    def _log_initialization_summary(self, initial_infected_count):
        """초기화 요약 로그 출력"""
        print(f"초기화 완료: {self.num_agents}명의 에이전트 로드 완료.")
        final_initial_vaccinated_count = int(self.cp.sum(self.agent_is_vaccinated))
        if final_initial_vaccinated_count > 0:
            coverage = final_initial_vaccinated_count / self.num_agents * 100 if self.num_agents > 0 else 0
            print(f"초기(Day 0) 백신 접종자 {final_initial_vaccinated_count}명이 설정되었습니다 ({coverage:.1f}%).")
        print(f"총 초기 감염자 {initial_infected_count}명이 설정되었습니다.")
        print(f"정책 데이터 로드 완료. 스케줄된 정책 변경: {len(self.policy_schedule)}건, 백신 캠페인: {len(self.vaccine_campaigns)}건")
        # 이동확대 모드 로드 요약
        print(f"이동확대 모드 스케줄 로드 완료: {len(self.travel_schedule)}건")

    def run(self):
        """
        시뮬레이션 메인 루프를 실행합니다.
        [수정] 24시간 루프 대신 3 스텝 루프로 변경합니다.
        """
        mode = "GPU" if self.IS_GPU_MODE else "CPU"
        print(f"\n{mode} 모드 시뮬레이션을 시작합니다...")

        daily_newly_confirmed_mask = self.cp.zeros(self.num_agents, dtype=self.cp.bool_)
        daily_newly_hospitalized_mask = self.cp.zeros(self.num_agents, dtype=self.cp.bool_)

        # Day 1부터 시작
        for day in range(1, self.config['simulation_days'] + 1):
            self.day = day
            start_time = time.time()

            # --- 일별 업데이트 시작 ---
            # 1. 정책, 캠페인, 이동확대 모드 적용 (마스크 정책 파라미터도 여기서 업데이트됨)
            apply_dynamic_policies(self)
            apply_vaccination_campaign(self)
            self._update_travel_mode()

            daily_movements = 0
            daily_newly_confirmed_mask[:] = False
            daily_newly_hospitalized_mask[:] = False

            # --- 스텝별 루프 시작 ([복원 #14] 4 스텝: 0=집, 1=transport, 2=목적지, 3=집) ---
            for step in range(4):
                self.step = step
                print(f"  - Day {self.day}, Step {self.step} 처리 중...", end='\r')

                # H1. 질병 진행 업데이트
                newly_i, newly_h, newly_r, newly_d = update_disease_progression(self)

                daily_newly_confirmed_mask |= newly_i
                daily_newly_hospitalized_mask |= newly_h

                # H2. 보건 정책 적용 (진단, 격리, 치료)
                apply_health_policies(self)

                # H3. 에이전트 이동 (이동확대 모드 반영됨)
                daily_movements += update_agents_locations(self)

                # H4. 감염 전파 (마스크 효과 반영됨)
                newly_e, infection_counts_by_loc = run_transmission(self)

                # H5. 로그 기록 및 모니터링
                log_step_stats(self, newly_e, newly_i, newly_h, newly_r, newly_d, infection_counts_by_loc)
                self._update_gpu_memory_log()

            # --- 스텝별 루프 종료 ---

            # 일별 로그 기록
            log_daily_regional_cases(self, daily_newly_confirmed_mask, daily_newly_hospitalized_mask)
            log_daily_movements(self, daily_movements)

            print(f"\nDay {self.day}: 시뮬레이션 완료. (소요 시간: {time.time() - start_time:.2f}초)")

        # --- 시뮬레이션 종료 후 처리 ---
        print("\n시뮬레이션이 성공적으로 종료되었습니다.")
        # [M3] 전염 페어 CSV 출력 (generation interval 분석용)
        save_gen_interval(self)
        if self.nvml_handle:
            print(f"최대 GPU 메모리 사용량: {self.max_gpu_memory_used / (1024 ** 2):.2f} MB")


# ==============================================================================
# ==============================================================================
# ===== SECTION 9: 시뮬레이션 실행기 (Runner 통합) =============================
# ==============================================================================
# ==============================================================================

# (이 섹션은 변경 없음)

# [수정] travel_config_file 인자 추가
def run_simulation(policy_name, policy_file, disease_name, simulation_days=150, force_gpu=True,
                   severity_rates_file=None, fatality_rates_file=None, travel_config_file=None, extra_config=None):
    """
    지정된 파라미터로 감염병 시뮬레이션을 설정하고 실행합니다.
    """
    print(f"\n{'=' * 60}\n[시뮬레이션 시작] 정책: {policy_name}, 질병: {disease_name}\n{'=' * 60}")

    # --- 1. 시뮬레이션 설정 구성 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir_name = f"outputs_simplified_{policy_name}_{timestamp}"

    simulation_config = {
        'simulation_days': simulation_days,
        'initial_infected_count': 4,
        # 기본 입력 파일 경로 설정 (원본 코드 기준 경로 유지)
        'initial_infections_file': './inputs/selected_sigungu_initial_infected_v2.xml',
        'agents_file': './initial_vals/agents_updated.xml',
        'places_file': './initial_vals/places.xml',
        # [수정] schedules_file은 더 이상 필수가 아님 (단순화 모델)
        'schedules_file': None,
        'disease_params_file': './inputs/disease_params.xml',
        'disease_name': disease_name,
        'policy_params_file': policy_file,
        'travel_config_file': travel_config_file,
        'output_dir': output_dir_name,
        'log_file': f'simulation_log_{policy_name}.csv',
        'force_gpu': force_gpu,
        'severity_rates_file': severity_rates_file,
        'fatality_rates_file': fatality_rates_file,
    }

    if extra_config:
        simulation_config.update(extra_config)

    # --- 2. 정책 및 이동확대 설정 파일 로드 ---
    # [수정] load_policy_params는 마스크 정책도 로드합니다.
    simulation_config['policy_params'] = load_policy_params(simulation_config['policy_params_file'])
    # 이동확대 설정 로드 (load_travel_config가 inputs/ 경로를 처리함)
    simulation_config['travel_params'] = load_travel_config(simulation_config['travel_config_file'])

    # --- 3. 실행 모드(CPU/GPU) 결정 ---
    print("--- 실행 모드 확인 시작 ---")
    backend_module = np
    is_gpu_mode = False

    if force_gpu:
        try:
            import cupy
            if cupy.is_available():
                backend_module, is_gpu_mode = (cupy, True)
                print("최종 모드: GPU (강제)")
            else:
                print("경고: GPU 모드가 강제되었으나 CuPy를 사용할 수 없습니다. CPU로 전환합니다.")
        except ImportError:
            print("=" * 60)
            # print("경고: CuPy 라이브러리가 없어 CPU 모드로 실행합니다.") # <-- 이 줄 제거 또는 주석 처리
            print("오류: CuPy 라이브러리 임포트 실패 (ImportError). CPU 모드로 실행합니다.")
            print("원인: 필요한 CUDA DLL (cuTENSOR, cuBLAS, cuDNN 등) 로드 실패 가능성 높음.")
            print("상세 오류 내역 (Traceback):")
            import traceback
            traceback.print_exc()  # 상세 오류 추적 정보 출력 (중요)
            print("=" * 60)
        except Exception as e:
            print(f"경고: CuPy 확인 중 예기치 않은 오류 발생 ({e}). CPU로 전환합니다.")
            import traceback
            traceback.print_exc()
    else:
        print("최종 모드: CPU (강제)")

    print("--- 실행 모드 확인 완료 ---")

    # [FL-EXT] 재현성: extra_config에 rng_seed가 있으면 NumPy/CuPy/stdlib random 시드 고정
    # (stdlib random 포함 필수 — _set_initial_infections_regional이 random.sample 사용)
    rng_seed = simulation_config.get('rng_seed')
    if rng_seed is not None:
        random.seed(int(rng_seed))
        np.random.seed(int(rng_seed))
        if is_gpu_mode:
            backend_module.random.seed(int(rng_seed))
        print(f"[FL-EXT] RNG seed 고정: {rng_seed}")

    # --- 4. 시뮬레이션 실행 및 결과 저장 ---
    try:
        sim = Simulation(simulation_config, backend_module, is_gpu_mode)
        sim.run()

        # 결과 처리
        output_dir_abs = os.path.abspath(simulation_config['output_dir'])
        log_file = os.path.join(output_dir_abs, simulation_config['log_file'])
        graph_output_file = os.path.join(output_dir_abs, f'summary_{policy_name}.png')

        save_daily_seihr_graph(log_file, graph_output_file)

        print(f"\n[시뮬레이션 성공] 결과가 '{output_dir_abs}'에 저장되었습니다.")
        return output_dir_abs

    except Exception as e:
        print(f"\n[실행 오류] 정책 '{policy_name}', 질병: {disease_name}' 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# ==============================================================================
# ===== SECTION 10: Rate Generator (rate_generator.py 통합) ====================
# ==============================================================================
# ==============================================================================
# (rate_generator.py의 함수들은 내용 변경이 없으므로 코드 블록을 합칩니다.)

# ==============================================================================
# (rate_generator.py 함수들 통합)
# ==============================================================================

# --- Helper & Fitting Functions ---
def prettify_xml(elem):
    """XML 요소를 예쁘게 포맷팅하여 문자열로 반환합니다."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    # 들여쓰기 적용 및 빈 줄 제거
    pretty_str = reparsed.toprettyxml(indent="    ", encoding="utf-8").decode('utf-8')
    return '\n'.join([line for line in pretty_str.split('\n') if line.strip()])


if curve_fit:
    def func_exponential(x, a, b):
        return a * np.exp(b * x)


    def func_logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))


    def func_gompertz(x, a, b, c):
        return a * np.exp(-b * np.exp(-c * x))


    FITTING_FUNCTIONS = {
        'exponential': func_exponential,
        'logistic': func_logistic,
        'gompertz': func_gompertz
    }
else:
    FITTING_FUNCTIONS = {}


# --- XML 파라미터 로딩 함수 ---
def load_multipliers_from_disease_params(file_path, disease_name):
    # ... (이 함수는 변경 없음) ...
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        disease_node = root.find(f".//Disease[@name='{disease_name}']")
        if disease_node is None:
            print(f"경고: '{file_path}'에서 '{disease_name}' 질병을 찾을 수 없습니다. 기본값(1.0)을 사용합니다.")
            return 1.0, 1.0

        sev_mult_node = disease_node.find('SeverityMultiplier')
        fat_mult_node = disease_node.find('FatalityMultiplier')

        sev_mult = float(sev_mult_node.text) if sev_mult_node is not None and sev_mult_node.text else 1.0
        fat_mult = float(fat_mult_node.text) if fat_mult_node is not None and fat_mult_node.text else 1.0

        print(f"'{file_path}'에서 '{disease_name}'의 배수를 로드했습니다: 중증화율({sev_mult}), 치명률({fat_mult})")
        return sev_mult, fat_mult
    except Exception as e:
        print(f"오류: '{file_path}' 파일 처리 중 문제가 발생했습니다 ({e}). 기본값(1.0)을 사용합니다.")
        return 1.0, 1.0


# --- Core Logic ---
def generate_modified_rates(coeffs_xml_path, multiplier, output_dir):
    # ... (이 함수는 변경 없음) ...
    """
    계수 파일과 배수를 사용하여 수정된 연령별 비율 XML을 생성합니다.
    """
    if not FITTING_FUNCTIONS:
        print("경고: scipy가 설치되지 않아 비율 생성을 건너뜁니다.")
        return None

    if not os.path.exists(coeffs_xml_path):
        print(f"경고: 계수 파일 '{coeffs_xml_path}'가 없습니다. 해당 비율 생성을 건너뜁니다.")
        return None

    try:
        tree = ET.parse(coeffs_xml_path)
    except (FileNotFoundError, ET.ParseError):
        print(f"경고: 계수 파일 '{coeffs_xml_path}' 로드/파싱 실패. 생성을 건너뜁니다.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    root = tree.getroot()
    rate_type = root.get('type')
    best_fit_func_name = root.get('best_fit_function')

    if not best_fit_func_name or best_fit_func_name not in FITTING_FUNCTIONS:
        return None

    func_el = root.find(best_fit_func_name.capitalize())
    if func_el is None: return None

    original_params = {k: float(v) for k, v in func_el.attrib.items() if k != 'r_squared'}

    print(f"\n--- 시나리오 기반 {rate_type.capitalize()} 비율 생성 ---")

    best_func = FITTING_FUNCTIONS[best_fit_func_name]
    # 파라미터 개수에 맞게 param_order 조정
    param_order = best_func.__code__.co_varnames[1:len(original_params) + 1]

    try:
        original_params_list = [original_params[p_name] for p_name in param_order]
    except KeyError:
        print(f"경고: '{coeffs_xml_path}'의 파라미터가 함수 정의와 일치하지 않습니다.")
        return None

    all_ages = np.arange(0, 101)
    base_rates = best_func(all_ages, *original_params_list)
    modified_rates = base_rates * multiplier

    rates_root = ET.Element("ScenarioRatesByAge", type=rate_type, multiplier=str(multiplier))
    rates_root.set("source_function", best_fit_func_name)

    for age, rate in zip(all_ages, modified_rates):
        rate_val = max(0, min(1.0, rate))
        ET.SubElement(rates_root, "Rate", age=str(age), value=f"{rate_val:.8f}")

    rates_path = os.path.join(output_dir, f"scenario_{rate_type}_rates_by_age.xml")
    with open(rates_path, "w", encoding="utf-8") as f:
        f.write(prettify_xml(rates_root))
    print(f"시나리오 기반 연령별 비율 파일 생성 완료: {rates_path}")
    return rates_path


# --- Main Function (Generator) ---
def generate_scenario_rates_for_disease(disease_name, base_input_dir, scenario_output_dir):
    # ... (이 함수는 변경 없음) ...
    """
    지정된 질병에 대한 중증/치명률 파일을 생성하고, 생성된 파일 경로를 반환합니다.
    """
    print(f"\n===== '{disease_name}'에 대한 중증/치명률 파일 생성 시작 =====")

    disease_params_file = os.path.join(base_input_dir, 'disease_params.xml')
    sev_multiplier, fat_multiplier = load_multipliers_from_disease_params(
        disease_params_file, disease_name
    )

    scenario_params = {"severity": sev_multiplier, "fatality": fat_multiplier}
    generated_files = {}

    for rate_name, mult in scenario_params.items():
        coeffs_file = f"fitted_{rate_name}_coefficients.xml"
        coeffs_path = os.path.join(base_input_dir, coeffs_file)

        # scenario_output_dir를 전달하여 파일 생성
        file_path = generate_modified_rates(coeffs_path, mult, scenario_output_dir)
        if file_path:
            generated_files[rate_name] = file_path

    print(f"===== '{disease_name}'에 대한 파일 생성 완료 =====\n")
    return generated_files


# ==============================================================================
# ==============================================================================
# ===== SECTION 11: Main Execution (run_main.py 통합) ==========================
# ==============================================================================
# ==============================================================================

# (이 섹션은 변경 없음)

def get_disease_names_from_xml(file_path):
    # ... (이 함수는 변경 없음) ...
    """
    disease_params.xml 파일을 읽어 정의된 모든 질병 이름을 리스트로 반환합니다.
    """
    if not os.path.exists(file_path):
        print(f"오류: 질병 파라미터 파일({file_path})을 찾을 수 없습니다.")
        return []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        disease_names = []
        for disease_node in root.findall('Disease'):
            name = disease_node.get('name')
            if name:
                disease_names.append(name)
        return disease_names
    except ET.ParseError as e:
        print(f"오류: XML 파일 파싱 중 문제 발생 ({file_path}): {e}")
        return []
    except Exception as e:
        print(f"오류: 질병 이름 로드 중 예기치 않은 오류 발생: {e}")
        return []


def main():
    """
    메인 실행 함수. 시나리오를 설정하고 시뮬레이션을 반복 실행합니다.
    """

    # --- 1. 설정 정의 ---
    DISEASE_PARAMS_FILE = './inputs/disease_params.xml'
    BASE_INPUT_DIR = './inputs'
    SIMULATION_DAYS = 150
    FORCE_GPU = True

    # 이동확대 모드 설정 파일 정의 (inputs 디렉토리 기준 파일명)
    # 이 파일명은 load_travel_config 함수에서 ./inputs/와 조합되어 사용됩니다.
    TRAVEL_CONFIG_FILE = 'travel_config.xml'

    # 실행할 정책 정의
    # policy.xml 파일에 마스크 정책이 포함되어 있어야 합니다.
    policies = [
        ('기본정책', './inputs/policy.xml'),
    ]

    # --- 2. 동적 시나리오 로드 ---
    print(f"'{DISEASE_PARAMS_FILE}'에서 사용 가능한 질병 목록을 로드합니다...")
    disease_names = get_disease_names_from_xml(DISEASE_PARAMS_FILE)

    if not disease_names:
        # 질병 목록 로드 실패 시 처리
        print("XML 파일에서 질병을 찾을 수 없거나 파일 로드에 실패했습니다. 실행을 종료합니다.")
        return

    # 시나리오 목록 생성
    scenarios = []
    for policy_name_base, policy_file in policies:
        for disease_name in disease_names:
            policy_name_full = f"{policy_name_base}_{disease_name}"
            scenarios.append((policy_name_full, policy_file, disease_name))

    # --- 3. 시뮬레이션 실행 ---
    total_scenarios = len(scenarios)
    print(f"총 {total_scenarios}개의 시나리오를 실행합니다.")
    start_time = time.time()

    for i, (policy_name, policy_file, disease_name) in enumerate(scenarios, 1):
        print(f"\n--- 시나리오 {i}/{total_scenarios} 실행: {policy_name} ({disease_name}) ---")

        # 시나리오별 입력 파일 디렉토리 경로 정의
        scenario_specific_input_dir = os.path.join(BASE_INPUT_DIR, 'scenarios', disease_name)

        # 시나리오별 디렉토리에 중증/치명률 파일 생성 (rate_generator 사용)
        generated_rate_files = generate_scenario_rates_for_disease(
            disease_name=disease_name,
            base_input_dir=BASE_INPUT_DIR,
            scenario_output_dir=scenario_specific_input_dir
        )

        # 파일 생성 성공 확인
        severity_file = generated_rate_files.get('severity')
        fatality_file = generated_rate_files.get('fatality')

        if 'severity' not in generated_rate_files or 'fatality' not in generated_rate_files:
            print(f"--- 시나리오 '{policy_name}' 경고: 중증/치명률 파일 생성에 실패했습니다. 기본값(0)으로 진행될 수 있습니다. ---")

        # 시뮬레이션 실행 (run_simulation 호출)
        # TRAVEL_CONFIG_FILE 전달
        result_dir = run_simulation(
            policy_name=policy_name,
            policy_file=policy_file,
            disease_name=disease_name,
            simulation_days=SIMULATION_DAYS,
            force_gpu=FORCE_GPU,
            severity_rates_file=severity_file,
            fatality_rates_file=fatality_file,
            travel_config_file=TRAVEL_CONFIG_FILE
        )

        if result_dir:
            print(f"--- 시나리오 '{policy_name}' 완료. 결과 저장됨: {result_dir} ---")
        else:
            print(f"--- 시나리오 '{policy_name}' 실패. ---")

    end_time = time.time()
    print(f"\n모든 시나리오 실행 완료. 총 소요 시간: {end_time - start_time:.2f}초")


if __name__ == "__main__":
    # 실행 전 확인사항:
    # 1. 필요한 입력 파일(agents_updated.xml, places.xml 등)이 ./initial_vals/ 에 있는지 확인하세요.
    #    (agents_updated.xml에는 이전 단계에서 추가된 preschool_id 속성이 포함되어야 합니다.)
    # 2. 설정 파일(policy.xml, disease_params.xml)이 ./inputs/ 에 있는지 확인하세요.
    # 3. 이동확대 모드를 사용하려면 inputs/travel_config.xml 파일을 준비하세요.
    main()