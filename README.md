# Term_Project_CICIoT2023_ML
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/299ac164-6efb-4554-b00b-643c78e9ace3" /><br/>
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/4e12e6b4-030f-4849-b2c5-a1a6d5ec8ca1" /><br/>

CICIoT2023 : 105개의 실제 사물인터넷 IoT 기기에서 33종의 최신 공격을 실행하여 수집한 데이터
https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset

<img width="770" height="412" alt="image" src="https://github.com/user-attachments/assets/4a148704-04ff-4899-b3d2-5273765dc2e2" /><br/>

- 소수 클래스 탐지 증가 : 그룹 분류에서 LR 모델에서 0.01에 불과했던 ‘Web’ 공격 점수를 0.56으로, ‘BruteForce’를 0.69로 크게 향상 시킴
- Dos/DDoS 혼동 문제 해결 : LR 모델에서 DoS와 DDoS를 혼동하던 문제를 RF 모델은 완벽히 해결 (점수 1.0)
- 치명적 공격 탐지 성공 : 전혀 탐지하지 못하던 SqlInjection과 XSS를 RF는 0.31, 0.13점으로 탐지하기 시작함.

<img width="744" height="371" alt="image" src="https://github.com/user-attachments/assets/91ee67f5-2edd-438e-a887-5cf74b6c6a4a" /><br/>

- 2-Class (이진 분류) : 두 모델 모두 99% 이상의 뛰어난 성능을 보임. 
  LightGBM이 XGBoost보다 아주 미세하게 우세함

- 8-Class (그룹 분류) : XGBoost가 정확도와 F1-Score 모두에서 LightGBM을 앞섬.
  LightGBM은 BruteForce, Web 등 데이터가 적은 소수 클래스를 분류하는데 있어 XGBoost 보다 성능이 떨어지는 경향을 보임

- 34-Class (세부 공격) : XGBoost는 클래스가 34개로 늘어나도 99%대의 높은 정확도를 유지하는 모습을 보이지만, 
  LightGBM은 정확도가 60%대로 급격히 무너짐. 


