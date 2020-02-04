#### CSRNet-Paddle

This is a simple and clean implemention of CVPR 2018 paper ["CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"]( https://arxiv.org/abs/1802.10062 ).

#### Requirement

1. PaddlePaddle 1.6.2

2. Python 3.6

#### Data Setup

1. Download ShanghaiTech Dataset from [Baidu](https://pan.baidu.com/s/1nBxiBFZV0naJp7t9A6HkFA)(code:y7qt)

#### Train

1. Download the parameters of the pre-training model VGG16 (vgg16.pkl) from [Baidu disk](https://pan.baidu.com/s/1_VQ2SOvAmsXLCah6x8PzZQ)(code:4vh3)
2. Run train.py

#### Testing

1. Run test.py for calculate MAE of test images .

#### Other notes

1. We trained the model and got 14.19  MAE  and 484.35 MSE at 471-th epoch on ShanghaiTech PartB.
2. For some reasons , the result is still a bit behind the paper. If you get better result , I hope you can tell me.

