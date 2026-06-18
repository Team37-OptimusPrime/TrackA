# Track A: 컴퓨터 자원 이용률과 에너지 소비량 사이의 비선형성 분석
### 1. 트랙 목표
기존의 선형적 에너지 분리의 부정확함을 증명하고, 비선형적 에너지 소비 패턴을 더 잘 설명할 수 있는 메트릭을 제시한다.
### 2. 제안 내용
1\)워크로드 특성에 따른 비선형적 전력 모델 수립 \
2\)GEMM 교차 실험을 통한 하드웨어 아키텍처 효과 분리
### 3. 주요 기능
| 기능 아이디 | 기능 제목 | 기능 설명 |
| :---:  | :---: | :---: |
| R1 | 실험 전 환경 자동 검증 | -워크로드 스크립트 존재 여부 및 Python 의존성 확인 <br/>-CPU와 GPU 통제 조건 검증</br>-계측 도구 가용성 확인|
|R2|실시간 매트릭 수집|-CPUdhk GPU 전력 수집</br>-CPU busy, GPU Utilization, /proc/stat 및 memory bandwidth utilization 측정|
|R3|실험 오케스트레이션|-온도 기반 idle 진입 판정</br>-CPU 중심 워크로드: taskset 코어 수 제한, 단일 노드 바인딩</br>-GPU 중심 워크로드: batch size 조절로 GPU SM 점유율 제어</br>-실험 사이클: Warm-up (측정 제외) -> 부하율별 워크로드 실행 -> Cool-down|
|R4|데이터 분석 및 시각화|총 6종의 그래프 생성|
### 4. How to build, install and test
#### ‼️환경 요구사항
- OS: Linux (Ubuntu 20.04 이상 권장)
- Python 3.8+
- CUDA가 지원되는 NVIDIA GPU
- nvidia-smi, perf, taskset 설치
```bash
#레포지토리 클론
git clone https://github.com/Team37-OptimusPrime/TrackA.git
cd TrackA

#Python 의존성 설치
pip install -r requirements.txt   # torch, pandas, matplotlib, scipy 등

#환경 검증 (R1)
python scripts/precheck.py

#전체 실험 실행 (R2+R3 통합 진입점)
python scripts/run_experiment.py

#분석 및 시각화 (R4)
python scripts/analyze_data.py

#결과 확인
ls figures_v4/   # 생성된 그래프 6종
ls results_v4/   # 원시 CSV 데이터
```
#### 4-1.부가기능: watchdog.sh
장시간 실험 중 GPU 하드웨어를 보호하기 위해 실행.
이 스크립트는 nvidia-smi를 5초 간격으로 폴링하여 GPU 온도(상한 85°C)와 전력 소비(상한 260 W)를 감시하고, 어느 한쪽이라도 임계값에 도달하면 즉시 pkill로 워크로드 프로세스를 종료한 뒤 로그에 원인을 기록하고 스스로 종료한다.
```bash
# 새 세션 시작
tmux new-session -d -s exp

# 왼쪽 창: 실험 스크립트
tmux send-keys -t exp 'python run_experiment.py --workload all' Enter

# 오른쪽 창 분할
tmux split-window -h -t exp

# 오른쪽 창: watchdog
tmux send-keys -t exp 'bash watchdog.sh' Enter

# 세션 접속 (모니터링)
tmux attach -t exp
```
### 5. 소프트웨어 구조도
<img src="https://github.com/user-attachments/assets/24a14447-f2ef-462d-bb3e-f417de731f16" >

### 6. 기술블로그 및 기타 참고</br>
[기술블로그(TBD)](https://us2xince.tistory.com/44)</br>
[캡스톤 디자인 보고서](https://github.com/Team37-OptimusPrime/optimusprime/blob/main/37%E1%84%90%E1%85%B5%E1%86%B7-OptimusPrime-2%E1%84%8E%E1%85%A1%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5-%E1%84%8C%E1%85%A6%E1%84%8E%E1%85%AE%E1%86%AF%E1%84%8C%E1%85%A1V2.%E1%84%80%E1%85%A1%E1%86%BC%E1%84%89%E1%85%B5%E1%84%8B%E1%85%A7%E1%86%AB.pdf)
