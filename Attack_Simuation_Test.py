import pandas as pd
import numpy as np
import joblib
import os
import time
import random
from datetime import datetime

# --- 1. 설정 및 파일 경로 ---
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_lgbm_model_34class.pkl')
LE_PATH = os.path.join(MODEL_DIR, 'label_encoder_34class.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight',
]


def load_resources():
    print("\n🔄 정밀 시뮬레이션 모드 로딩 중...")
    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일 없음")
        return None, None, None

    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, le, scaler


def generate_precise_traffic(attack_type):
    """
    AI가 헷갈리지 않도록 공격/정상 특징을 극단적으로 부여
    """
    data = {col: 0.0 for col in X_columns}  # 실수형 초기화

    # 공통 베이스 (노이즈)
    data['flow_duration'] = np.random.uniform(0.05, 2.0)
    data['Header_Length'] = np.random.uniform(50, 150)
    data['Tot size'] = np.random.uniform(60, 500)

    if attack_type == 'Benign':
        # [정상]: 모든 수치가 낮고 평범함
        data['Protocol Type'] = 6.0  # TCP
        data['TCP'] = 1.0
        data['HTTP'] = np.random.choice([0.0, 1.0], p=[0.8, 0.2])  # 가끔 웹 사용
        data['Rate'] = np.random.uniform(1, 20)  # 아주 낮은 전송률
        data['Srate'] = np.random.uniform(1, 20)
        data['ack_count'] = np.random.uniform(1, 10)  # 정상적인 ACK 교환
        data['Duration'] = np.random.uniform(0.1, 5.0)
        data['Weight'] = np.random.uniform(1, 10)  # 정상 가중치

    elif attack_type == 'DDoS_UDP':
        # [DDoS UDP]: UDP + 압도적인 전송량
        data['Protocol Type'] = 17.0  # UDP
        data['UDP'] = 1.0
        data['Rate'] = np.random.uniform(50000, 100000)  # 미친 속도
        data['Srate'] = np.random.uniform(50000, 100000)
        data['Tot size'] = np.random.uniform(500, 1400)  # 꽉 찬 패킷
        data['IAT'] = np.random.uniform(0.0001, 0.001)  # 패킷 간격 매우 짧음

    elif attack_type == 'DDoS_TCP_SYN':
        # [DDoS SYN Flood]: TCP + SYN 플래그 도배
        data['Protocol Type'] = 6.0
        data['TCP'] = 1.0
        data['syn_flag_number'] = 1.0  # 핵심 특징
        data['syn_count'] = np.random.uniform(100, 500)
        data['Rate'] = np.random.uniform(10000, 50000)

    elif attack_type == 'Mirai':
        # [Mirai]: UDP 위주 + 특정 패턴
        data['Protocol Type'] = 17.0
        data['UDP'] = 1.0
        data['Rate'] = np.random.uniform(500, 2000)
        data['Weight'] = 244.0  # Mirai가 자주 보이는 특정 가중치 흉내
        data['Radius'] = np.random.uniform(100, 300)

    elif attack_type == 'Web_XSS':
        # [웹 해킹]: HTTP + 긴 페이로드(Max size) + 긴 Duration
        data['Protocol Type'] = 6.0
        data['TCP'] = 1.0
        data['HTTP'] = 1.0
        data['Max'] = np.random.uniform(1000, 8000)  # 비정상적으로 큰 패킷 (스크립트 삽입 시도)
        data['Duration'] = np.random.uniform(30, 120)  # 연결 안 끊음
        data['Rate'] = np.random.uniform(5, 50)  # 속도는 느림

    elif attack_type == 'Recon_Scan':
        # [포트 스캔]: RST/FIN 플래그 + 빠른 연결 시도/종료
        data['Protocol Type'] = 6.0
        data['TCP'] = 1.0
        data['rst_flag_number'] = 1.0  # 찔러보고 끊기
        data['fin_flag_number'] = 1.0
        data['rst_count'] = np.random.uniform(50, 200)
        data['Rate'] = np.random.uniform(50, 200)

    elif attack_type == 'BruteForce':
        # [무차별 대입]: SSH/Telnet + 높은 빈도의 패킷 수(Number)
        data['Protocol Type'] = 6.0
        data['TCP'] = 1.0
        data['SSH'] = 1.0
        data['Number'] = np.random.uniform(50, 200)  # 짧은 시간 동안 많은 시도
        data['Rate'] = np.random.uniform(20, 100)

    return pd.DataFrame([data], columns=X_columns), attack_type


def run_simulation():
    model, le, scaler = load_resources()
    if model is None: return

    print("\n" + "=" * 80)
    print("      🛡️ IoT 지능형 보안 관제 시스템 (High Precision Mode)")
    print("      (정상 트래픽과 공격 트래픽의 특징을 명확히 구분합니다)")
    print("=" * 80)
    time.sleep(1)

    # 테스트 비율: 정상(50%), 공격(50%)
    scenarios = ['Benign'] * 5 + ['DDoS_UDP', 'DDoS_TCP_SYN', 'Mirai', 'Web_XSS', 'Recon_Scan', 'BruteForce']

    packet_id = 1

    try:
        while True:
            # 1. 시나리오 생성
            scenario = random.choice(scenarios)
            traffic_df, true_scenario = generate_precise_traffic(scenario)

            # 2. 전처리
            if scaler:
                X_input = pd.DataFrame(scaler.transform(traffic_df), columns=X_columns)
            else:
                X_input = traffic_df

            # 3. 예측
            y_pred_enc = model.predict(X_input)
            y_pred_prob = model.predict_proba(X_input)
            confidence = np.max(y_pred_prob) * 100
            label = le.inverse_transform(y_pred_enc)[0]

            # 4. 결과 출력 포맷팅
            now = datetime.now().strftime("%H:%M:%S")

            # (A) 정상이 정상으로 탐지됨 -> 초록색
            if label == 'BenignTraffic' and scenario == 'Benign':
                log = f"[{now}] ID:{packet_id:04d} | 🟢 정상 패킷 통과 (Safe)         | 시나리오: {scenario:<12} | 확신도: {confidence:.1f}%"

            # (B) 공격이 공격으로 탐지됨 -> 빨간색
            elif label != 'BenignTraffic' and scenario != 'Benign':
                log = f"[{now}] ID:{packet_id:04d} | 🚨 공격 탐지! [{label:<20}] | 시나리오: {scenario:<12} | 확신도: {confidence:.1f}%"

            # (C) 오탐지 (정상인데 공격으로, 공격인데 정상으로) -> 노란색 경고
            else:
                log = f"[{now}] ID:{packet_id:04d} | ⚠️ 오탐지 주의 [{label:<20}] | 시나리오: {scenario:<12} | 확신도: {confidence:.1f}%"

            print(log)

            # 5. 대응 메시지
            if label != 'BenignTraffic':
                if confidence > 80:
                    if 'DDoS' in label:
                        print(f"      ㄴ 🛡️ [System] 대역폭 차단 수행 (DDoS 대응)")
                    elif 'Mirai' in label:
                        print(f"      ㄴ 🛡️ [System] 해당 IoT 디바이스 네트워크 격리")
                    elif 'Web' in label or 'XSS' in label:
                        print(f"      ㄴ 🛡️ [System] 악성 페이로드 차단 (WAF 작동)")
                else:
                    print(f"      ㄴ 👁️ [System] 의심 활동 모니터링 중 (확신도 낮음)")

            time.sleep(random.uniform(0.5, 1.2))
            packet_id += 1

    except KeyboardInterrupt:
        print("\n🛑 시스템 종료.")


if __name__ == "__main__":
    run_simulation()