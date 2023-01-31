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

$$ f(x, y) = x^2 - y^2 $$  

그럼 어떻게 x, y 좌표를 움직여서 최대한 빨리 공의 높이를 최소화 할 수 있을까요? 


## 1. (Stochastic) Gradient Descent

가장 먼저 떠올릴 수 있는 방법은 해당 좌표에서 공을 굴려 보는 것입니다. 공은 각 위치에서 가장 경사가 낮은 방향으로 굴러가겠죠. 이게 Gradient Descent의 아이디어입니다. 

<center>
<img src="/assets/gradientdescent.png">
</center>

함수를 변수로 미분한 값은 해당 함수의 기울기를 의미하며, 해당 함수를 가장 빠르게 증가시킬 수 있는 방향을 의미합니다.  
그럼 반대로 해당 함수를 가장 빠르게 감소할 수 있는 방향은 -미분값이겠죠. 그래서 Gradient descent는 -미분값을 이용하여 해당 함수의 최소값을 찾아가는 방법입니다.


그럼 실제로 위의 예제에서 Gradient Descent를 적용해보겠습니다.
우리가 최소화 하고싶은 x, y 좌표에 대한 공의 함수를 미분하면 다음과 같습니다. 

$$ \frac{\partial f}{\partial x} = 2x $$
$$ \frac{\partial f}{\partial y} = -2y $$

우리의 시작점은 x=2, y=0.001이며, 미분값은 x=4, y=-0.002가 되겠죠.  
그럼 이제 x, y 좌표를 각각 -미분값만큼 이동시킨다면 x는 -4만큼, y는 0.002만큼 이동하게 됩니다. 

하지만 이렇게 되면 이동한 x좌표는 -2가 되고, y좌표는 0.003가 되는데, 해당 좌표에서 한 번 더 Gradient descent를 한다면 x는 다시 2가 되고, y는 0.005가 되는 상황이 발생합니다. 
위 함수를 시각화 한 사진을 봤을 때, x는 0에 가까워져야 하는데, x가 너무 크게 뛰어다녀서 0에 가까워지지 못하는 상황이 발생하는 것이죠. 

이를 해결하는 장치가 Learning rate입니다. Learning rate는 Gradient descent를 할 때, 미분값을 얼마나 반영할지를 결정하는 값입니다.  
쉽게 말하면 미분 값에 상수를 곱하여 이동할 거리를 결정하는 것이죠. Learning rate는 0.001같이 매우 작은 값을 주로 사용합니다. 

그럼 Learning rate를 사용해 위의 예제를 다시 적용해보겠습니다. 
Learning rate를 0.001로 설정하고, x, y 좌표를 각각 -미분값만큼 이동시키면 x는 -0.004만큼, y는 0.000002만큼 이동하게 됩니다.
그러면 x는 1.996, y는 0.000999가 되는데, x가 0에 가까워지고, y는 0에 가까워지는 것을 볼 수 있습니다.  

위에서 말로 설명한 것을 수식으로 표현하면 다음과 같습니다.  

$$ w_t = w_{t-1} - \alpha \frac{\partial f}{\partial w} $$  

여기서 w는 x, y 좌표를 의미하고, t는 반복 횟수를 의미합니다. 즉 T번 반복하면서 w를 업데이트 하는 것이죠.  

이제 위의 수식을 코드로 구현해보겠습니다. 

```python
import jax.numpy as jnp

def return_lr(lr, step):
    if callable(lr):
        return lr(step)
    else:
        return lr


class Optimizer:
    def __init__(self):
        self.step = 0

    def update(self, grads: Optional[jnp.ndarray] = None):
        self.step += 1

    def __call__(self, params, grads):
        update_val = self.update(grads)
        return params + update_val


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        return -lr * grads
```

우선 Learning rate를 받은 후, learning rate가 callable한 값이면 step에 따른 learning rate를 반환하고, 아니면 그냥 learning rate를 반환하는 함수 return_lr을 만들었습니다.  
그 뒤 앞으로 계속 Optimizer를 구현하는데 기반이 될 Optimizer 클래스를 만들었습니다. 어떤 Optimizer든 step by step으로 update되며, scheduler를 사용하기 위해 step 정보가 필요하기 때문에 step을 저장하는 변수를 만들었습니다. 

실제로 구현하고 싶은 SGD 클래스는 Optimizer를 상속받아 만들어집니다. 실제로 update를 하는 부분은 Optimizer의 __call__함수가 담당하기에, 우리는 어떤 값으로 Update를 할지 정해주기 위해 update 함수를 오버라이딩 하면 됩니다.  
SGD의 Update rule은 우리가 위에서 봤던 수식과 같습니다.  

실제 위 코드로 실행한 최적화 결과는 다음과 같습니다.  


<center>
<img src="/assets/sgd.gif">
</center>


## Momentum SGD

Momentum SGD는 SGD에 momentum을 추가한 것입니다. 생각해보면 실제로 언덕에서 공을 굴린다면 공이 일정한 방향으로 계속 굴러갈 때 더 빠르게 굴러가는 것을 볼 수 있습니다. 이 때문에 한 방향으로 계속 Gradient descent가 이뤄진다면, SGD보다 더 빨리 수렴할 수 있습니다.  
또한 Momentum SGD는 SGD보다 더 안정적으로 수렴합니다.  

<center>
<img src="/assets/momentum.png">
</center>

위 예시를 보면 Gradient descent는 현 시점의 기울기만을 보기 때문에, 작은 턱에 걸려서 Global minima로 수렴하지 못합니다. 하지만 Momentum SGD는 현 시점의 기울기와 이전 시점의 기울기를 고려하기 때문에, 턱을 넘어서 Global minima로 수렴할 수 있겠죠. 
당연히 noise가 있는 상황에서도 이전 움직임을 어느정도유지하려 하기 때문에 더 robust한 특성도 보이죠.  

Momentum SGD를 수식으로 표현하면 다음과 같습니다.

$$ v_{t+1} = \beta v_t + \alpha \frac{\partial f}{\partial w} $$
$$ x_{t+1} = x_t - v_{t+1} $$

여기서 v는 velocity, 현재의 관성을 의미하며, $\beta$는 momentum을 의미합니다. beta가 클수록 이전 시점을 더 많이 고려하게 됩니다. 
사실 EMA에서는 과거시점에는 $\beta$를, 현 시점에는 $(1-\beta)$를 곱해주나, Momentum SGD에서는 과거 시점에만 $\beta$를 곱해주는 것을 볼 수 있습니다.

위의 수식을 사용해 Momentum SGD를 구현하면 다음과 같습니다.  

```python
class MomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        self.velocity = self.momentum * self.velocity - lr * grads
        return self.velocity
```

이번에도 update 함수를 오버라이딩 하며 update rule만 수정했습니다. 

실제로 Momentum SGD를 사용해 최적화를 수행한 결과는 다음과 같습니다.

<center>
<img src="/assets/msgd.gif">
</center>

경사로를 SGD보다 훨씬 더 빨리 타고 내려가는 것이 보입니다.  

## Nesterov Momentum SGD 

작성중