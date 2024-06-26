## 강화학습이란

- 프로그램이 특정 환경에서 어떤 행동을 취함으로써 보상을 최대화하는 방법을 학습하는 것이다. 
- 이러한 학습은 프로그램이 행동을 선택하고 그 결과를 관찰함으로써 이루어지며 시행착오를 통해 보상을 최대화하는 행동방식을 찾는 것이 목표이다.

## 학습 방식에 따른 특징

### 비지도학습
- **특징:** 분류되지 않은 데이터나 숨겨진 구조를 찾는 데 사용된다.
- **예시:** 동물의 신체 특성이나 사회적 경향과 같이 유사한 특징에 따라 데이터를 군집화할 수 있다..

### 지도학습
- **특징:** 주어진 입력에 대해 레이블을 적용하는 방식으로 학습된다.
- **문제점:**
  - 정답 라벨을 만들어야 한다.
  - 라벨링은 번거롭고 비용이 많이 들며, 실시간으로 변화하는 환경에서 어려울 수 있다.

### 강화학습
- **특징:** 최적의 결과를 생성하는 최상의 행동 시퀀스를 찾는다.
- **장점:** 보상 신호를 통해 학습되기 때문에 정답 라벨이 필요 없다. 이로써 비용을 절감하고 실시간으로 변화하는 환경에서도 적용 가능하다.

## 강화학습 방식의 문제점과 해결책

- **문제점:**
  - 복잡한 문제에서는 상태 공간의 크기가 너무 커져 테이블로 상태를 나타내는 것이 어렵다.
- **해결책:**
  - 딥러닝을 적용하여 상태 공간의 크기에 관계없이 학습이 가능하도록 한다.
  - 따라서 경험하지 못한 상태에 대해서도 예측이 가능해진다.

## 강화학습 적용 예시
[AI Learns to Park - Deep Reinforcement Learning](https://youtu.be/VMp6pq6_QjI?si=fKTz2nvY67-zLYeG) <br>
[AI Learns Parallel Parking - Deep Reinforcement Learning](https://youtu.be/MlFZjLkEIEw?si=PvT05yPy-A6ctv8-)
