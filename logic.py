"""
logic.py - SPC 통계 계산 및 머신러닝(Random Forest) 예측 로직 모듈
스마트 공정 품질 관리 시스템 (Hybrid Ver 1.0)
"""

import math
import statistics
import random
from typing import List, Tuple, Dict, Any, Optional

# scikit-learn 및 numpy 라이브러리 가용성 확인
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    np = None


# ==========================================
# 1. SPC (통계적 공정 관리) 로직
# ==========================================

def calculate_spc_limits(
    dimensions: List[float],
    target_dim: float = 10.0,
    default_sigma: float = 0.1,
) -> Tuple[float, float, float, float]:
    """
    수집된 치수(dimension) 리스트를 기반으로 통계량 및 관리한계선(UCL, LCL)을 산출합니다.
    
    공식:
    - 중심선 / 평균 (CL / Mean, X_bar)
    - 표준편차 (Std Dev, sigma)
    - UCL (Upper Control Limit) = Mean + 3 * sigma
    - LCL (Lower Control Limit) = Mean - 3 * sigma
    
    Returns:
        (mean, std_dev, ucl, lcl)
    """
    if not dimensions or len(dimensions) < 2:
        mean = target_dim if not dimensions else float(dimensions[0])
        std_dev = default_sigma
    else:
        mean = float(statistics.mean(dimensions))
        try:
            std_dev = float(statistics.stdev(dimensions))
        except Exception:
            std_dev = default_sigma
        if std_dev == 0.0:
            std_dev = default_sigma

    ucl = mean + (3.0 * std_dev)
    lcl = mean - (3.0 * std_dev)
    
    return round(mean, 4), round(std_dev, 4), round(ucl, 4), round(lcl, 4)


def evaluate_spc(
    dimension: float,
    ucl: float,
    lcl: float,
) -> Tuple[bool, str, str]:
    """
    측정된 치수가 관리상하한선(UCL/LCL) 내에 있는지 평가합니다.
    
    Returns:
        (is_violated, status_code, message)
        - is_violated: True면 관리 이탈(비정상), False면 정상
        - status_code: 'NORMAL' 또는 'OUT_OF_CONTROL'
        - message: 상세 판정 사유 안내
    """
    if dimension > ucl:
        return True, "OUT_OF_CONTROL", f"치수({dimension:.3f}mm)가 관리상한선 UCL({ucl:.3f}mm)을 초과했습니다."
    elif dimension < lcl:
        return True, "OUT_OF_CONTROL", f"치수({dimension:.3f}mm)가 관리하한선 LCL({lcl:.3f}mm) 미만입니다."
    else:
        return False, "NORMAL", f"치수({dimension:.3f}mm)가 정상 관리한계 내에 위치합니다."


# ==========================================
# 2. 머신러닝 (Random Forest) 및 특성 중요도 분석
# ==========================================

class DefectPredictor:
    """
    공정 환경 데이터(온도, 습도, 치수)를 바탕으로 불량 발생 확률을 예측하고,
    Random Forest의 Feature Importance를 계산하여 지능형 원인 분석을 제공하는 모델 래퍼.
    """

    FEATURE_NAMES = ["온도", "습도", "치수"]

    def __init__(self, n_estimators: int = 50, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.is_sklearn = SKLEARN_AVAILABLE
        self.model = None
        self.feature_importances: Dict[str, float] = {"온도": 0.25, "습도": 0.15, "치수": 0.60}

        self._initialize_and_train_dummy_model()

    def _generate_synthetic_data(self, n_samples: int = 600) -> Tuple[List[List[float]], List[int]]:
        """
        제조 공정 특성을 반영한 합성 학습 데이터셋 생성:
        - 공정 기준 조건: 온도 23℃, 습도 50%, 치수 10.0mm
        - 조건을 벗어날수록 불량 확률(is_defect=1)이 증가하도록 구성
        """
        rng = random.Random(self.random_state)
        X = []
        y = []

        for _ in range(n_samples):
            temp = rng.uniform(15.0, 32.0)
            hum = rng.uniform(25.0, 80.0)
            dim = rng.gauss(10.0, 0.25)

            defect_score = 0.0

            # 온도 요인 (23도 기준 편차)
            temp_dev = abs(temp - 23.0)
            if temp_dev > 4.0:
                defect_score += (temp_dev - 4.0) * 0.15

            # 습도 요인 (50% 기준 편차)
            hum_dev = abs(hum - 50.0)
            if hum_dev > 15.0:
                defect_score += (hum_dev - 15.0) * 0.02

            # 치수 요인 (10.0mm 기준 편차, 품질에 가장 큰 영향)
            dim_dev = abs(dim - 10.0)
            if dim_dev > 0.2:
                defect_score += (dim_dev - 0.2) * 2.8

            # 무작위 공정 노이즈
            defect_score += rng.uniform(-0.05, 0.05)

            defect_prob = max(0.0, min(1.0, defect_score))
            is_defect = 1 if defect_prob >= 0.35 else 0

            X.append([round(temp, 2), round(hum, 2), round(dim, 3)])
            y.append(is_defect)

        return X, y

    def _initialize_and_train_dummy_model(self) -> None:
        """더미 데이터셋을 생성하고 Random Forest 모델을 사전 학습시킵니다."""
        X_train, y_train = self._generate_synthetic_data(n_samples=600)

        if self.is_sklearn and RandomForestClassifier is not None:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                random_state=self.random_state,
            )
            self.model.fit(X_train, y_train)
            
            # 실제 Random Forest 모델로부터 특성 중요도(Feature Importance) 도출
            importances = self.model.feature_importances_
            total = sum(importances) if sum(importances) > 0 else 1.0
            self.feature_importances = {
                name: round(float(imp / total), 4)
                for name, imp in zip(self.FEATURE_NAMES, importances)
            }
        else:
            self.model = None
            self.feature_importances = {"온도": 0.25, "습도": 0.15, "치수": 0.60}

    def predict(
        self, temperature: float, humidity: float, dimension: float
    ) -> Tuple[float, int, Dict[str, float], str, str]:
        """
        입력된 공정 파라미터(온도, 습도, 치수)에 따른 불량 확률 및 Feature Importance 기반 원인 분석 수행.

        Returns:
            (defect_prob_percent, is_defect, feature_importances, top_factor, cause_comment)
            - defect_prob_percent: 불량 발생 확률 (%)
            - is_defect: 불량 판정 (0: 정상, 1: 불량)
            - feature_importances: 모델의 특성 중요도 딕셔너리 {'온도': 0.xx, '습도': 0.xx, '치수': 0.xx}
            - top_factor: 가장 큰 영향을 미친 요인 명칭 ('온도' / '습도' / '치수')
            - cause_comment: '현재 상태에서 불량 확률에 가장 큰 영향을 미친 요인은 [치수]입니다.' 형식의 코멘트
        """
        if self.is_sklearn and self.model is not None:
            features = [[temperature, humidity, dimension]]
            probs = self.model.predict_proba(features)[0]
            classes = list(self.model.classes_)
            if 1 in classes:
                defect_idx = classes.index(1)
                defect_prob = float(probs[defect_idx])
            else:
                defect_prob = 0.0
            defect_prob_percent = round(defect_prob * 100.0, 2)
        else:
            score = 0.0
            if abs(temperature - 23.0) > 4.0:
                score += (abs(temperature - 23.0) - 4.0) * 12.0
            if abs(humidity - 50.0) > 15.0:
                score += (abs(humidity - 50.0) - 15.0) * 1.5
            if abs(dimension - 10.0) > 0.2:
                score += (abs(dimension - 10.0) - 0.2) * 150.0
            prob = max(1.0, min(99.0, score + 3.0))
            defect_prob_percent = round(prob, 2)

        is_defect = 1 if defect_prob_percent >= 50.0 else 0

        # --- 지능형 원인 분석 (현재 상태에서 불량 확률에 가장 큰 영향을 미친 요인 산출) ---
        temp_dev = abs(temperature - 23.0) / 4.0
        hum_dev = abs(humidity - 50.0) / 15.0
        dim_dev = abs(dimension - 10.0) / 0.2

        fi = self.feature_importances
        impact_scores = {
            "온도": temp_dev * fi.get("온도", 0.25),
            "습도": hum_dev * fi.get("습도", 0.15),
            "치수": dim_dev * fi.get("치수", 0.60),
        }

        # 만약 모든 편차가 거의 0인 아주 정상 상태인 경우 기본 Feature Importance가 가장 높은 요인 선정
        max_val = max(impact_scores.values())
        if max_val < 0.001:
            top_factor = max(fi, key=fi.get)
        else:
            top_factor = max(impact_scores, key=impact_scores.get)

        # 요구사항 지정 코멘트 형식: '현재 상태에서 불량 확률에 가장 큰 영향을 미친 요인은 [온도/습도/치수 중 1개]입니다.'
        cause_comment = f"현재 상태에서 불량 확률에 가장 큰 영향을 미친 요인은 [{top_factor}]입니다."

        return defect_prob_percent, is_defect, fi, top_factor, cause_comment


# 싱글톤 예측기 인스턴스
default_predictor = DefectPredictor()


def predict_defect(
    temperature: float, humidity: float, dimension: float
) -> Tuple[float, int, Dict[str, float], str, str]:
    """
    공정 입력값(온도, 습도, 치수)으로부터 불량 예측 결과 및 특성 중요도 원인 분석 반환
    """
    return default_predictor.predict(temperature, humidity, dimension)


if __name__ == "__main__":
    print("=== [logic.py] Feature Importance & 지능형 원인 분석 테스트 ===")
    test_cases = [
        (23.0, 50.0, 10.01, "정상 케이스"),
        (33.0, 52.0, 10.01, "고온 이상 케이스"),
        (23.0, 85.0, 10.02, "고습도 이상 케이스"),
        (23.2, 49.0, 10.45, "치수 이탈 이상 케이스"),
    ]

    for temp, hum, dim, desc in test_cases:
        prob, is_def, fi, factor, comment = predict_defect(temp, hum, dim)
        print(f"\n[{desc}] 입력: {temp}℃, {hum}%, {dim}mm")
        print(f"- 불량확률: {prob}% ({'불량' if is_def else '정상'})")
        print(f"- Feature Importance: {fi}")
        print(f"- 최상위 영향 요인: {factor}")
        print(f"- 코멘트: {comment}")
