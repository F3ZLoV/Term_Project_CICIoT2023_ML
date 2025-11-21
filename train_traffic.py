# 0. 공통 라이브러리 및 머신러닝 라이브러리 임포트
import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm

# (추가) 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# (추가) 계층적 샘플링을 위한 라이브러리
from sklearn.model_selection import train_test_split

# # 3, 4단계 머신러닝 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, \
    confusion_matrix

# (선택) 한글 폰트 설정 (시각화 시)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
except:
    pass  # Windows가 아닐 경우 무시
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 경고 메시지 무시
warnings.filterwarnings('ignore')

print("--- 0. 라이브러리 임포트 완료 ---")

# --- 1. 데이터 로드 및 전처리 설정 ---

# 🚨 CICIOT2023 CSV 파일들이 있는 디렉터리 경로
DATASET_DIRECTORY = 'CICIoT2023/'

# ⭐️ (중요) 메모리 오류 해결을 위한 샘플링 비율
# 0.1 = 각 파일에서 10%의 데이터만 계층적 샘플링으로 추출합니다.
# 메모리가 여전히 부족하면 0.05 (5%) 등으로 더 낮춰보세요.
SAMPLING_RATIO = 0.1

try:
    all_files = [k for k in os.listdir(DATASET_DIRECTORY) if k.endswith('.csv')]
    all_files.sort()
    print(f"\n--- 1. 데이터 로드 ---")
    print(f"총 {len(all_files)}개 CSV 파일 발견.")
    print(f"설정된 샘플링 비율: {SAMPLING_RATIO * 100}%")
except FileNotFoundError:
    print(f"오류: '{DATASET_DIRECTORY}' 경로를 찾을 수 없습니다.")
    exit()

# --- 2. 데이터 전처리 (피처/레이블 정의 및 매핑) ---

# # 4. X (특성)와 y (타겟) 분리 (컬럼 이름 정의)
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
y_column = 'label'

print(f"\n--- 2. 데이터 전처리 ---")
print(f"X (특성) 컬럼 {len(X_columns)}개 정의 완료.")
print(f"y (타겟) 컬럼 '{y_column}' 정의 완료.")

# 레이블 매핑 딕셔너리 정의
dict_8_classes = {
    'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS', 'DoS-UDP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
    'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
    'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon',
    'DNS_Spoofing': 'Spoofing', 'MITM-ArpSpoofing': 'Spoofing',
    'BenignTraffic': 'Benign', 'BrowserHijacking': 'Web', 'Backdoor_Malware': 'Web',
    'XSS': 'Web', 'Uploading_Attack': 'Web', 'SqlInjection': 'Web',
    'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'
}
dict_2_classes = {'BenignTraffic': 'Benign'}

print("2-Class / 8-Class 레이블 매핑 딕셔너리 정의 완료.")


# (개선) ⭐️ 샘플링 기능이 추가된 데이터 로드/전처리 함수
def load_and_preprocess(files, description, sample_ratio):
    """파일 목록을 읽어 샘플링하고, 하나의 DataFrame으로 합친 후 전처리/스케일링합니다."""
    sampled_dfs = []
    print(f"\n{description} 데이터 로드 및 샘플링(비율 {sample_ratio * 100}%) 중...")

    for f in tqdm(files):
        file_path = os.path.join(DATASET_DIRECTORY, f)
        try:
            df = pd.read_csv(file_path, low_memory=False)

            # 1. NaN/Infinity 값을 0으로 대체 (스케일링 전)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            # 2. 계층적 샘플링 (Stratified Sampling)
            try:
                _, df_sample = train_test_split(
                    df,
                    test_size=sample_ratio,  # test_size를 샘플링 비율로 사용
                    stratify=df[y_column],  # 레이블 비율 유지
                    random_state=42
                )
            except ValueError:
                df_sample = df.sample(frac=sample_ratio, random_state=42)

            sampled_dfs.append(df_sample)

        except Exception as e:
            print(f"파일 {f} 처리 중 오류: {e}")

    # 샘플링된 모든 DataFrame을 하나로 합침
    full_df = pd.concat(sampled_dfs, ignore_index=True)

    # 3. X(특성)와 y(타겟) 분리
    X = full_df[X_columns]
    y_raw = full_df[y_column]

    # 4. 스케일링 적용 (scaler는 3단계에서 미리 fit 되어 있어야 함)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_columns)

    return X_scaled_df, y_raw


print("\n--- 2. 데이터 전처리 완료 ---")

# --- 3. 모델 훈련 시작 ---

# # 1. 훈련/테스트 데이터 분할 (파일 기준 80:20)
split_index = int(len(all_files) * 0.8)
training_files = all_files[:split_index]
test_files = all_files[split_index:]

print(f"\n--- 3. 모델 훈련 ---")
print(f"[데이터 분할 현황]")
print(f"훈련용 파일 {len(training_files)}개, 테스트용 파일 {len(test_files)}개.")

# # 3. 데이터 스케일링 (학습)
print("\n[데이터 스케일링 완료 (StandardScaler)]")
print("훈련 파일들로 StandardScaler 학습(fitting) 중 (시간이 오래 걸릴 수 있습니다)...")
scaler = StandardScaler()
for train_set_file in tqdm(training_files):
    file_path = os.path.join(DATASET_DIRECTORY, train_set_file)
    try:
        df_chunk = pd.read_csv(file_path, usecols=X_columns, low_memory=False)
        df_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_chunk.fillna(0, inplace=True)
        scaler.partial_fit(df_chunk)  # 점진적 학습
    except Exception as e:
        print(f"파일 {train_set_file} 처리 중 오류: {e}")
del df_chunk
print("StandardScaler 학습 완료.")

# (개선) 샘플링된 훈련/테스트 데이터 로드
try:
    X_train, y_train_raw = load_and_preprocess(training_files, "훈련", sample_ratio=SAMPLING_RATIO)
    X_test, y_test_raw = load_and_preprocess(test_files, "테스트", sample_ratio=SAMPLING_RATIO)

    print(f"\n[데이터 샘플링 후 Shape]")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_raw.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_raw.shape}")

except MemoryError:
    print(f"\n[오류] 샘플링(현재 {SAMPLING_RATIO * 100}%) 후에도 메모리 부족.")
    print("스크립트 상단의 SAMPLING_RATIO 값을 더 낮춰서 (예: 0.05) 다시 시도해주세요.")
    exit()

# 시나리오별 y (타겟) 데이터 생성
y_train_34 = y_train_raw
y_test_34 = y_test_raw
y_train_8 = y_train_raw.map(dict_8_classes).fillna('Benign')
y_test_8 = y_test_raw.map(dict_8_classes).fillna('Benign')
y_train_2 = y_train_raw.map(dict_2_classes).fillna('Attack')
y_test_2 = y_test_raw.map(dict_2_classes).fillna('Attack')

# --- 1-bis. 📊 상세한 탐색적 데이터 분석 (EDA) (샘플링 데이터 활용) ---
print(f"\n--- 1-bis. 상세 EDA (샘플링된 훈련 데이터 기준) ---")

# (추가) 1-1. (시각화) 2-Class (Attack/Benign) 분포 파이 차트
try:
    plt.figure(figsize=(8, 8))
    label_counts = y_train_2.value_counts()
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.2f%%',
            startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title(f'샘플링된 훈련 데이터 레이블 분포 (2-Class, {SAMPLING_RATIO * 100}% Sample)')
    plt.legend()
    plt.savefig("eda_1_pie_2_classes.png")
    print("[시각화] 'eda_1_pie_2_classes.png' 저장 완료")
except Exception as e:
    print(f"시각화 1 (파이 차트) 오류: {e}")

# (수정) 1-2. (시각화) 34-Class 세부 레이블 분포 (상위 20개)
try:
    plt.figure(figsize=(12, 10))
    # 원본 34개 클래스 중 상위 20개만
    top_20_labels = y_train_34.value_counts().nlargest(20)
    sns.barplot(y=top_20_labels.index, x=top_20_labels.values)
    plt.title(f'샘플링된 훈련 데이터 세부 레이블 분포 (Top 20 / 34-Class)')
    plt.xlabel('데이터 수')
    plt.ylabel('공격 유형 (원본)')
    plt.xscale('log')  # 수량 차이가 크므로 log 스케일
    plt.tight_layout()
    plt.savefig("eda_2_bar_top20_labels.png")
    print("[시각화] 'eda_2_bar_top20_labels.png' 저장 완료 (로그 스케일)")
except Exception as e:
    print(f"시각화 2 (세부 레이블) 오류: {e}")

# (추가) 1-3. (시각화) 주요 특성 분포 (Benign vs Attack) - Box Plot
try:
    features_to_plot = ['flow_duration', 'Rate', 'Tot sum', 'AVG']
    print(f"[시각화] 주요 특성 {features_to_plot} 박스플롯 저장 중...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    temp_df = X_train[features_to_plot].copy()
    # (주의) 스케일링된 데이터(X_train)는 이미 평균 0, 표준편차 1로 변환됨
    # 원본 분포를 보려면 load_and_preprocess에서 X(스케일링 전)를 반환해야 함
    # 여기서는 스케일링된 데이터의 분포를 비교
    temp_df['label'] = y_train_2.values

    for i, feature in enumerate(features_to_plot):
        sns.boxplot(data=temp_df, x='label', y=feature, ax=axes[i], showfliers=False)  # Outlier 제외
        axes[i].set_title(f"'{feature}' 분포 (Benign vs Attack) - Scaled Data")
        # (참고) Y축 스케일이 매우 작을 수 있음 (이미 스케일링됨)
        # axes[i].set_yscale('log') # Box plot은 log scale 적용이 까다로울 수 있음

    plt.tight_layout()
    plt.savefig("eda_3_feature_boxplot.png")
    print("[시각화] 'eda_3_feature_boxplot.png' 저장 완료")
    del temp_df
except Exception as e:
    print(f"시각화 3 (특성 박스플롯) 오류: {e}")

# 1-4. (시각화) 주요 특성 간 상관관계 히트맵 (유지)
try:
    print("[시각화] 주요 특성 상관관계 히트맵 저장 중...")
    corr_features = X_columns[:15]
    corr_matrix = X_train[corr_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.1f', cmap='coolwarm_r')
    plt.title('주요 특성 상관관계 히트맵 (상위 15개) - Scaled Data')
    plt.tight_layout()
    plt.savefig("eda_4_correlation_heatmap.png")
    print("[시각화] 'eda_4_correlation_heatmap.png' 저장 완료")
    del corr_matrix
except Exception as e:
    print(f"시각화 4 (상관관계) 오류: {e}")

# # 4. 모델 생성 및 훈련 (EDA 완료 후 진행)
print("\n[모델 생성 및 훈련]")
# (max_iter=1000: 수렴 경고 방지)

#%% 모델 1 (2-Class)
print("LogisticRegression (2 classes) 훈련 시작...")
model_2 = LogisticRegression(n_jobs=-1, max_iter=1000)
model_2.fit(X_train, y_train_2)

#%% 모델 2 (8-Class)
print("LogisticRegression (8 classes) 훈련 시작...")
model_8 = LogisticRegression(n_jobs=-1, max_iter=1000)
model_8.fit(X_train, y_train_8)

#%% 모델 3 (34-Class)
print("LogisticRegression (34 classes) 훈련 시작...")
model_34 = LogisticRegression(n_jobs=-1, max_iter=1000)
model_34.fit(X_train, y_train_34)

print("\n--- 3. 모델 훈련 완료 ---")

# --- 4. 모델 평가 및 해석 ---
print(f"\n--- 4. 모델 평가 및 해석 ---")

# # 2. 예측 수행
print("\n[테스트 데이터(X_test)에 대한 예측 완료]")
y_pred_2 = model_2.predict(X_test)
y_pred_8 = model_8.predict(X_test)
y_pred_34 = model_34.predict(X_test)


# (추가) Confusion Matrix 시각화 함수
def plot_confusion_matrix(y_true, y_pred, labels, model_name, filename):
    """Confusion Matrix를 시각화하고 저장하는 함수"""
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        show_annot = len(labels) <= 10

        plt.figure(figsize=(max(10, len(labels) * 0.8), max(8, len(labels) * 0.6)))
        sns.heatmap(cm, annot=show_annot, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filename)
        print(f"[시각화] '{filename}' 저장 완료")
    except Exception as e:
        print(f"시각화 (혼동 행렬: {model_name}) 오류: {e}")


# # 3. 성능 지표 출력 및 해석
def print_evaluation_metrics(model_name, y_true, y_pred, average_mode='macro', pos_label=None):
    """PPT의 평가 지표처럼 성능을 출력하는 함수"""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n##### {model_name} #####")
    print(f"[정확도(Accuracy)]: {accuracy:.4f} (약 {accuracy * 100:.2f}%)")

    print(f"\n[분류 리포트(Classification Report)]: (average='{average_mode}')")
    # zero_division=0: 샘플이 없는 클래스에 대해 0으로 처리
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    # binary 모드일 때 pos_label을 전달
    if average_mode == 'binary' and pos_label is not None:
        print(
            f"* Precision (정밀도 - {average_mode}): {precision_score(y_true, y_pred, average=average_mode, pos_label=pos_label, zero_division=0):.4f}")
        print(f"* Recall (재현율 - {average_mode}): {recall_score(y_true, y_pred, average=average_mode, pos_label=pos_label, zero_division=0):.4f}")
        print(f"* F1-score (조화 평균 - {average_mode}): {f1_score(y_true, y_pred, average=average_mode, pos_label=pos_label, zero_division=0):.4f}")
    else:
        print(
            f"* Precision (정밀도 - {average_mode}): {precision_score(y_true, y_pred, average=average_mode, zero_division=0):.4f}")
        print(f"* Recall (재현율 - {average_mode}): {recall_score(y_true, y_pred, average=average_mode, zero_division=0):.4f}")
        print(f"* F1-score (조화 평균 - {average_mode}): {f1_score(y_true, y_pred, average=average_mode, zero_division=0):.4f}")


# 2진 분류 평가 (Attack/Benign)
print_evaluation_metrics("LogisticRegression (2 classes)", y_test_2, y_pred_2, average_mode='binary')
plot_confusion_matrix(y_test_2, y_pred_2, model_2.classes_,
                      "2 classes", "result_images/eval_cm_2_classes.png")

# 8종 분류 평가
print_evaluation_metrics("LogisticRegression (8 classes)", y_test_8, y_pred_8, average_mode='macro')
plot_confusion_matrix(y_test_8, y_pred_8, model_8.classes_,
                      "8 classes", "result_images/eval_cm_8_classes.png")

# 34종 분류 평가 (Confusion Matrix는 너무 커서 비활성화)
print_evaluation_metrics("LogisticRegression (34 classes)", y_test_34, y_pred_34, average_mode='macro')

# --- 4-1. 보너스 – 특성이 공격/정상에 미치는 영향력 ---
print(f"\n--- 4-1. 보너스: 특성 영향력 (2-Class 모델) ---")
try:
    target_class_index = list(model_2.classes_).index('Attack')
    print(f"('{model_2.classes_[target_class_index]}' 클래스 기준 계수)")

    coefficients = model_2.coef_[target_class_index]
    coef_df = pd.DataFrame({'Feature': X_columns, 'Coefficient': coefficients})
    coef_df['abs_coef'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False)

    print("\n[특성(Feature)이 'Attack' 탐지에 미치는 영향력 (계수)]")
    print(coef_df[['Feature', 'Coefficient']].head(10))

    # 보너스 시각화 (상위/하위 10개)
    top_n = 10
    bottom_n = 10
    top_features = coef_df.head(top_n)
    bottom_features = coef_df.tail(bottom_n).sort_values(by='Coefficient', ascending=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=top_features)
    plt.title(f"'Attack' 확률을 높이는 상위 {top_n}개 특성 (Scaled Data)")
    plt.savefig("eval_feature_importance_positive.png", bbox_inches='tight')
    print(f"\n[시각화] 'eval_feature_importance_positive.png' 저장 완료")

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=bottom_features)
    plt.title(f"'Attack' 확률을 낮추는 (Benign에 가까운) 상위 {bottom_n}개 특성 (Scaled Data)")
    plt.savefig("eval_feature_importance_negative.png", bbox_inches='tight')
    print(f"[시각화] 'eval_feature_importance_negative.png' 저장 완료")

except Exception as e:
    print(f"시각화 (특성 영향력) 오류: {e}")

print("\n--- 모든 작업 완료 ---")