---
title: "An Overview of MLP-based models (1)"
layout: post
mathjax: true
---

 MLP-based model은 CNN model, Transformer-based vision model을 뒤이어 최근 컴퓨터 비전 커뮤니티에서 중요하게 연구되는 구조입니다.   

 이번 포스트에서는 MLP-based model에 대해 설명하기 보다는 컴퓨터 비전 분야에서 주류 모델들이 어떻게 발전해왔고, 왜 CNN-based model, Tranformer-based vision model 등 먼 길을 돌아 MLP-based model이 다시 주류가 되었는지를 설명하고자 합니다.   




## Classic Multi-Layer Perceptron
<center>
<img src="/assets/mlp.png">  
</center>

<center>
<em>Fig. 1 MLP architecture</em> <sup id="a1">[1]</sup>
</center>

 가장 먼저 현재 딥러닝의 시초가 되는 원시적인 MLP를 떠올려 봅시다.  MLP와 가장 많이 사용된 MNIST를 함께 예제로 들면, 28 * 28 * 1 사이즈의 이미지를 tabular 데이터처럼 flatten한 후 여러 층을 거쳐 정보를 추출하여 분류기에 정보를 넘겨줍니다.  사실, 기존의 MLP만 사용하더라도 MNIST같이 간단한 예제에서는 95% 이상의 꽤 괜찮은 성능을 보입니다. 하지만 성능은 둘째 치더라도, MLP는 translation invariance를 갖지 못한다는 큰 단점이 있죠. 

<center>
<img src="/assets/translation_invariance.gif">  
</center>

<center>
<em>Fig. 2 Translation Invariance</em> <sup id="a2">[2]</sup>
</center>

 Translation invariance는 특정한 패턴이 다른 위치에 등장하더라도 모델이 등장 위치에 상관 없이 일정한 예측을 할 수 있는 능력입니다. 그러나 MLP는 공간에 대한 정보를 모델에서 감지할 수 없기에 이 능력을 갖지 못했으며, [Fig. 2]처럼 특정 패턴이 이미지 내에서 움직임에 따라 모델의 logits가 크게 요동침을 볼 수 있습니다. 후술할 CNN은 이러한 능력을 갖고 있었기에, 컴퓨터 비전을 견인하는 첫 번째 모델이 될 수 있었습니다.

## Convolutional Neural Network

<center>
<img src="/assets/cnn.png">  
</center>

<center>
<em>Fig. 3 CNN architecture</em> <sup id="a3">[3]</sup>
</center>

 CNN은 이미지에 특정 범위내의 패턴을 탐지하는 필터를 씌워서 정보를 얻습니다. 당연히 짐작할 수 있듯이 더 큰 필터를 씌우는 것이 더 넓은 범위의 패턴을 감지할 수 있기에 더 좋은 성능을 보장하지만 필터의 크기가 커질 때 마다 파라미터 숫자와 계산량이 지수적으로 증가하기에 무작정 필터 크기를 키우기에는 한계가 있었죠. 

 재미있는 점은 convolution layer를 연속해서 사용하는 것은 곧 필터의 크기를 키우는 것과 같은 효과를 내지만, 파라미터 수는 오히려 적다는 것입니다. 즉 3 x 3 크기의 필터를 가진 convolution layer를 2번 거치는 것은 더 적은 연산량으로 5 x 5필터 사이즈를 가진 convolution layer 1개를 거치는 것과 같은 범위를 볼 수 있다는 것이죠. 당연히 MLP에서 여러 층을 쌓는 것 처럼 더 복잡한 함수를 mapping하는 장점도 있기에, 초창기 CNN에 대한 연구는 필터의 사이즈를 적당하게 조절하면서 안정적으로 많은 레이어를 쌓는 것에 집중 했습니다. Weight initialization이나 intermediate normalization에 대한 연구가 이뤄짐에 따라 쌓아갈 수 있는 레이어의 개수는 점차 증가하게 되었고, 마침내 ResNet이 등장함으로써 매우 깊은 신경망을 구현할 수 있게 되어 진정한 “딥러닝"이라 부를 수 있는 시대가 도래하게 됩니다.

<center>
<img src="/assets/resnetmodel.jpg">  
</center>

<center>
<em>Fig. 4 ResNet(36-layers)의 구조</em>
</center>

 레이어들이 계속해서 곱해지는것이 아닌, 계속 더해지게 만든다는 ResNet의 컨셉은 간단하지만 매우 효과적이었습니다. 152층이란 깊이를 안정적으로 학습한 ResNet에 이어, identity mapping과 pre-activation을 사용해 성능 저하없이 1000층을 안정적으로 학습해낸 ResNet의 개선작까지 등장하게 되면서 이제 신경망 깊이의 제한은 사실상 사라진 것이나 다름 없었습니다.

 깊이의 제한이 사라졌다는 것은 곧 모델에서 사용할 수 있는 자원의 현실적인 제한이 사라졌다는 것과 다름없었기에,  같은 자원에서 더 효율적으로 정보를 추출할 수 있는 구조를 찾게 되었습니다. Dense block, SE block, depth-wise/point-wise convolution, separable convolution 및 EfficientNet의 자원 분배의 수식화 등을 예로 들 수 있습니다.

## Transformer

 잠시 주제를 바꿔 자연어 처리 분야로 넘어가 보죠. CNN을 적극적으로 사용한 컴퓨터 비전 분야와 다르게, sequential data를 처리해야 했던 자연어 처리 분야에서는 RNN류 모델을 주로 사용했었습니다. 하지만 RNN은 학습이 불안정하며 먼 시점의 데이터를 처리하기 힘들다는 단점이 있었기에 이를 개선한 transformer에게 자리를 내줍니다.

<center>
<img src="/assets/transformer.png">  
</center>

<center>
<em>Fig. 5 Transformer architecture</em>
</center>

 Transformer의 특징은 “여러 시점의 데이터를 한번에 처리한다" 로 요약할 수 있습니다. 순차적으로 하나씩 시점을 처리해야 했던 RNN에 비해 더 안정적이며 빠른 속도로 연산할 수 있는 것이죠. 그러면서도 각 시점간의 연관성을 모델이 처리할 수 있게 multi-head self attention을 활용 했습니다. Multi-head self attention은 각 시점간의 연관성을 모델이 파악할 수 있도록 feature map에서 Query, Key, Value를 추출한 후 Query와 Key를 사용해 각 시점의 연관성을 나타내는 attention map을 만든 뒤, 이를 다시 Value에 곱해 다른 시점의 정보를 효율적으로 처리합니다. 또한 multi-head라는 접두사가 붙은 것에서 알 수 있듯이, 각각의 Query, Key, Value를 N개의 head로 분리 한 후 앞에서 설명한 self attention 연산을 하여, 같은 연산량에서도 좀 더 다양한 정보를 추출할 수 있도록 했습니다.

 본 포스트는 transformer를 위한 포스트가 아니기에 여기에선 컨셉만 간단히 설명했지만, 중요한 개념이기에 jalammar같이 잘 설명된 블로그를 꼭 한번 보시는 것을 추천드립니다.

## Transformer-based Vision model(Vision Transformer)

<center>
<img src="/assets/visiontransformer.png">  
</center>

<center>
<em>Fig. 6 ViT architecture</em>
</center>

 Transformer를 컴퓨터 비전 분야로 가져온 첫 시도가 바로 Vision Transformer(ViT)입니다. 왜 굳이 기존의 CNN를 대신 sequential data를 처리하기 위해 만들어진 transformer를 컴퓨터 비전으로 끌어 왔을 까요? 그 이유는 CNN의 구조적 한계에 있습니다.

<center>
<img src="/assets/cnn_vs_transformer.png">  
</center>

<center>
<em>Fig. 7 Limit of CNN </em>
</center>

 앞서 말했듯이 CNN은 이미지에서 나타나는 특정 패턴을 탐지함이 목적이었습니다. 그럼 여기서 더 나아가 [Fig. 7]의 예제처럼 패턴간의 상대적인 위치도 파악해서 좌/우가 다른 이미지임을 알 수가 있을까요? 알 수 있습니다. 하지만 RNN에서 먼 시점의 데이터를 처리하기 위해 모델의 capacity를 늘렸던 것 처럼, 필터의 사이즈가 충분히 커질 때 까지 모델의 capacity를 늘려야 가능합니다. RNN에서 보였던 문제를 CNN에서도 보이고 있다 할 수 있죠. 이에 기존 transformer에서 한번에 긴 시점을 처리 했던 것 처럼, vision transformer에서는 각각의 이미지 조각(image patch)들을 시점처럼 활용하여 모든 이미지를 한번에 처리하고자 합니다.

<center>
<img src="/assets/inductivebias_example.png">  
</center>

<center>
<em>Fig. 7 Inductive bias </em> <sup>[2]</sup>
</center>

 하지만 transformer의 약한 inductive bias 때문에 CNN보다 항상 좋은 결과를 보이진 않았습니다. Inductive bias는 데이터가 아닌 모델 자체에서 발생하는 bias입니다. 예를 들어 image restoration에서 주로 사용하는 방식인 loss term의 변화로 모델이 semantic information에 집중하게 만든다던지, edge information에 집중하게 만드는 등의 방법을 생각하면 이해하기 쉽습니다. 물론 위의 경우는 모델의 loss에서 inductive bias가 발생하지만, CNN 같은 경우는 필터 내에서 보이는 정보는 인접한 픽셀에 위치한다는 모델 구조 자체에서 기인하는 inductive bias가 발생하죠. 이러한 inductive bias는 머신러닝의 가장 기초에서 배우는 bias-variance trade-off와 비슷하게 동작합니다. 적당한 inductive bias는 데이터셋의 variance가 크더라도 잘 수렴할 수 있게 해주지만, 너무 큰 inductive bias는 일반화 하기 힘든 결과를 보이겠죠. 

<center>
<img src="/assets/transformervscnnperformance.jpg">  
</center>

<center>
<em>Fig. 8 Performance evaluation </em>
</center>

 Inductive bias는 [Fig. 8]에서 보이는 것 처럼 vision transformer가 더 많은 데이터셋을 사용 할 수록 CNN보다 성능이 좋아지는 이유를 설명해줍니다. 학습에 필요한 데이터 셋이 적을 때는 CNN의 inductive bias 가 더 높은 성능을 보장해주지만, 학습에 사용되는 데이터 셋이 많아질수록 vision transformer의 global receptive field가 CNN의 inductive bias를 이기게 되는 것이죠. 마침 구글에서 vision transformer의 약한 inductive bias가 문제되지 않을 엄청난 규모의 데이터셋인 JFT-300M을 가지고 있었기에 vision transformer의 진정한 성능을 끌어낼 수 있었습니다.

## MLP-based Model

 잠시 스크롤을 올려 [Fig. 6]를 다시 보고 올까요? Vision transformer의 블록은 크게 두 phase로 나뉩니다. 첫 번째 phase는 multi-head self attention을 활용해 이미지의 spatial information을 처리하며, 두 번째 phase는 MLP(1x1 convolution)을 활용하여 channel information를 처리합니다. 이후의 vision transformer variants의 블록을 봐도 spatial information과 channel information을 나눠서 처리하며, 각각의 spatial/channel 정보를 처리하는 방식만 바꾸는 것을 볼 수 있습니다. 

<center>
<img src="/assets/mlpmixer.jpg">  
</center>

<center>
<em>Fig. 9 MLP-Mixer architecture</em>
</center>

 구글의 연구자들은 굳이 computational cost가 엄청 비싼 multi-head self attention을 사용해서 spatial information을 처리해야 하는가에 대해 의문을 던지며 새로운 모델 구조를 제안했으니, 그 모델이 바로 MLP-Mixer, MLP-based model의 시초입니다.




<b>2편에서 계속 ...</b>


<b id="f1">[1]:</b> https://www.pikpng.com/pngvi/himiwJb_mlp-network-architecture-for-mnist-multilayer-perceptron-clipart/ [↩](#a1)  
<b id="f1">[2]:</b> https://samiraabnar.github.io/articles/2020-05/indist [↩](#a2)  
<b id="f1">[3]:</b> https://www.researchgate.net/figure/An-example-of-CNN-architecture_fig1_320748406 [↩](#a3)  
