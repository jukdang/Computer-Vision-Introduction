### RANSAC algorithm with SIFT

1. RANSAC (Random Sample Consensus)
  - 데이터에서 임의의 sample을 추출해 각각 inlier, outlier를 계산하고, 이과정을 여러번 수행해 가장 inlier가 많은 경우가 우리가 구하고자 하는 model이 되는 것이다.
  
2. SIFT (Scale Invariant Feature Transform)
  - keypoint가 되는 특징점에서 window를 잡고 각 pixel들을 일정한 단위로 묶어 grandient orientation에 대한 angle histogram 형태로 저장하면 하나의 vector로 해당 점의 특징을 표현할 수 있다.
  - 동일한 이미지에서 비슷한 keypoint가 여럿 있을수 있고 matching되는 point들을 비교해 가장 잘되었을때와 두번째 잘되었을때를 비교해 차이가 거의 없다면 해당 점은 특징점이 아니게 된다.
