---
title: "Optimizer 기초부터 알아보기"
layout: post
mathjax: true
---

 이번 포스트는 딥러닝 모델을 최적화 하기 위한 Optimizer에 대해 SGD부터 Adabelief까지 발전 과정, 그리고 수식 및 구현에 대해 알아볼 것이며, 
실제로 Optimizer가 최적화를 진행하는 과정을 시각화 하여 이해하기 쉽게 작성할 것입니다.

## 문제 정의

<center>
<img src="/assets/Saddle.png">
</center>

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

$$ w_t = w_{t-1} - \alpha dw $$  

여기서 w는 x, y 좌표를 의미하고, t는 반복 횟수를 의미하며, dw는 gradient를 의미합니다. 
즉 T번 반복하면서 w를 -alpha * gradient로 업데이트 하는 것이죠.  

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

Momentum SGD는 SGD에 momentum을 추가한 것입니다. 생각해보면 실제로 언덕에서 공을 굴린다면 공이 일정한 방향으로 계속 굴러갈 때 관성 때문에 더 빠르게 굴러가는 것을 볼 수 있습니다. 
Momentum SGD는 이 원리를 적용한 것입니다. 

<center>
<img src="/assets/momentum.png">
</center>

위 예시를 보면 Gradient descent는 현 시점의 기울기만을 보기 때문에, 작은 턱에 걸려서 Global minima로 수렴하지 못합니다. 
하지만 Momentum SGD는 관성때문에 이전 움직임을 어느정도 따라가려 하기에, 턱을 넘어서 Global minima로 수렴할 수 있겠죠. 
당연히 현 시점의 gradient에 noise가 있는 상황에서도 이전 움직임을 어느정도유지하려 하기 때문에 더 robust한 특성도 보일 수 있습니다.  

Momentum SGD를 수식으로 표현하면 다음과 같습니다.

$$ v_t = \beta v_{t-1} - \alpha dw $$  
$$ w_t = w_{t-1} + v_t $$

여기서 v는 velocity로 t 시점의 관성을 의미하며, $$\beta$$는 이전 관성의 영향력 의미합니다. $$\beta$$가 클수록 이전 시점을 더 많이 고려하게 됩니다. 
사실 EMA에서는 과거시점에 $$\beta$$를, 현 시점에는 $$(1-\beta)$$를 곱해주나, Momentum SGD에서는 과거 시점에만 $$\beta$$를 곱해주는 것을 볼 수 있습니다.

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

실제로 Momentum SGD를 사용해 최적화를 수행한 결과는 다음과 같습니다.

<center>
<img src="/assets/msgd.gif">
</center>

일정한 방향으로 최적화 될 때 경사로를 SGD보다 훨씬 더 빨리 타고 내려가는 것이 보입니다. 엄청난 발전이네요.  
하지만 관성 때문에 오히려 반대편 경사로를 살짝 타고 올라갔다가 다시 내려오는 것 또한 보입니다. 즉 관성 때문에 일정 지점에서는 Overshooting을 하는 경향을 보입니다. 
그럼 이런 문제를 어떻게 해결할 수 있을까요?  

## Nesterov Momentum SGD(Nesterov Accelerated Gradient descent; NAG)

NAG는 위에서 말한 Overshooting 문제를 "생각을 하고 도박을 하는 것 보다 도박을 하고 생각하는 것이 일반적으로 더 좋다" 라는 아이디어로 해결합니다.  

<center>
<img src="/assets/nag_concept.jpg">
</center>

NAG는 이전 시점의 관성으로 먼저 이동한 뒤(도박을 한 뒤), 해당 시점에서 gradient를 계산하여 이동합니다. Momentum SGD(좌)와 NAG(우)의 도달점 화살표가 살짝 차이나는걸 볼 수 있죠. 
왜 이런 방식이 Overshooting을 해결할 수 있을까요?

<center>
<img src="/assets/nag.png">
</center>

위 예시에서 Momentum SGD(위)에 비해 NAG(아래)가 Overshooting에서 더 빨리 회복 하는 것을 볼 수 있습니다.
Momentum SGD는 최저점에 거의 도달했을 때, 관성 + gradient로 업데이트를 하기에 바로 최저점을 매우 크게 지나쳐버리지만, 
NAG는 관성으로 먼저 반대방향으로 이동한 뒤 반대방향에서 최저점으로 가는 gradient를 계산하기에 Overshooting을 줄이게 되는 것이죠.

NAG를 수식으로 표현하면 다음과 같습니다.  

$$ v_t = \beta v_{t-1} - \alpha d(w - v_{t-1}) $$  
$$ w_t = w_{t-1} + v_t $$

$$ d(w - v_{t-1}) $$는 관성으로 이동한 뒤의 gradient를 의미합니다.  

위의 수식을 코드로 구현하면 다음과 같습니다.  


```python
class NesterovMomentumSGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-3, momentum: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0.

    def update(self, grads: Optional[jnp.ndarray] = None):
        lr = return_lr(self.learning_rate, self.step)
        super().update()

        self.velocity = self.momentum * self.velocity - lr * grads
        return self.momentum * self.velocity - (lr * grads)
```

수식과 코드를 정확히 보신 분들은 느꼈겠지만, 뭔가 이상하지 않나요? velocity로 이동한 뒤 미분을 계산하지도 않고,
update rule이 위 수식과 조금 다르죠. 위 구현체는 Keras에서 구현한 것과 같기에 틀린 구현도 아닌데, 왜 수식과 다를까요?  

### NAG 수식과 Keras 구현의 차이

 사실 Keras의 구현이 NAG 수식과 "다른"것은 아닙니다. 위 코드의 마지막 줄을 수식으로 풀어쓴다면 
 
$$ w_t = w_{t-1} + \beta v_t - \alpha dw $$  
가 됩니다. 이를 조금 변형해보면  

$$
\begin{linenomath*}
\begin{align}
w_t &= w_{t-1} + \beta v_t - \alpha dw \\
    &= w_{t-1} + \beta v_t - \alpha dw + \beta v_{t-1} - \beta v_{t-1} \\ 
    &= w_{t-1} - \beta v_{t-1} + \beta v_{t-1} - \alpha dw + \beta v_t \\
    &= w_{t-1} - \beta v_{t-1} + (\beta v_{t-1} - \alpha dw) + \beta v_t 
\end{align}
\end{linenomath*}
$$  
가 됩니다. 다시 정리해보죠. 1번에서는 단순히 $$v_t$$를 치환했고, 2번에서는 $$ \beta v_{t-1} - \beta v_{t-1} = 0 $$을 이용했고,
3번은 단순히 순서를 바꾼 것이며, 4번은 보기 쉽게 괄호를 친게 끝입니다. 

<center>
<img src="/assets/nagkeras.png">
</center>  

자 그럼 위 사진을 보면, 기존 NAG의 경우, 0번에서 1번으로 점프를 한 뒤 1번에서 2번으로 미분을 계산하여 이동하게 됩니다.  
하지만 Keras의 NAG 경우는 1번에서 시작해서 미분을 계산($$\alpha dw$$)합니다. 그럼 위 수식의 6번에서 
$$ - \beta v_{t-1} $$는 1번에서 0번으로 Momentum인 갈색 화살표를 빼는 것이며, $$ + (\beta v_{t-1} - \alpha dw) $$ 는 0에서 2로 가는 과정을 합친 녹색 화살표죠. 
마지막으로 $$ \beta v_t $$는 2에서 3, 즉 다음 미분 계산을 위해 현재의 momentum을 미리 더해 놓는 것입니다. NAG는 0->1->2 순으로 이동하나,
Keras의 NAG는 1->0->2 순서로 이동한 뒤 다음 시점을 위해 2->3을 더한 것이란 의미죠. 이 때문에 Weight에 veloicty를 더하는 추가적인 작업을 하지 않고도 NAG와 거의 같은 결과를 얻을 수 있습니다.  


실제로 NAG로 최적화를 수행한 결과는 다음과 같습니다.  

<center>
<img src="/assets/nag.gif">
</center>  


## AdaGrad(Adaptive Gradient)

작성중
