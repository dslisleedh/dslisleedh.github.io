---
title: "Optimizer 기초부터 알아보기"
layout: post
mathjax: true
---

 이번 포스트는 딥러닝 모델을 최적화 하기 위한 Optimizer에 대해 SGD부터 Adabelief까지 발전 과정, 그리고 수식 및 구현에 대해 알아볼 것이며, 
실제로 Optimizer가 최적화를 진행하는 과정을 시각화 하여 이해하기 쉽게 작성할 것입니다. 

<center>
<img src="/assets/Saddle.png">
</center>>

우선 Optimizer로 해결하고 싶은 문제상황은 위 사진과 같습니다. x(2.), y(0.001)에 위치한 공의 좌표를 움직이면서 해당 좌표에 위치한 공의 높이를 최소화 하고 싶은 상황입니다.  
해당 좌표에 대한 공의 높이 함수는 다음과 같습니다.

$$ f(x, y) = x**2 + y**2 $$  

그럼 어떻게 x, y 좌표를 움직여서 최대한 빨리 공의 높이를 최소화 할 수 있을까요? 


## 1. (Stochastic) Gradient Descent

가장 먼저 떠올릴 수 있는 방법은 해당 좌표에서 공을 굴려 보는 것입니다. 공은 각 위치에서 가장 경사가 낮은 방향으로 굴러가겠죠. 이게 Gradient Descent의 아이디어입니다.

