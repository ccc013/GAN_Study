# GAN_Study
学习GAN的笔记和代码



## The original code address

[tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)



### Papers

简单总结一些看过或者待看的论文，相应的代码，学习笔记，文章等。

### First paper for GAN

- [x] Generative Adversarial Networks [[Paper]](https://arxiv.org/abs/1406.2661)[[Code]](https://github.com/goodfeli/adversarial)[[paper_note]](https://mp.weixin.qq.com/s?__biz=MzU5MDY5OTI5MA==&mid=2247483732&idx=1&sn=99cb91edf6fb6da3c7d62132c40b0f62&chksm=fe3b0f21c94c8637a8335998c3fc9d0adf1ac7dea332c2bd45e63707eac6acad8d84c1b3d16d&token=985117826&lang=zh_CN#rd)


参考文章：

- [深度 | 生成对抗网络初学入门：一文读懂GAN的基本原理（附资源）](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650730721&idx=2&sn=95b97b80188f507c409f4c72bd0a2767&chksm=871b349fb06cbd891771f72d77563f77986afc9b144f42c8232db44c7c56c1d2bc019458c4e4&scene=21#wechat_redirect)
- [beginners-review-of-gan-architectures](https://sigmoidal.io/beginners-review-of-gan-architectures/)
- [干货 | 深入浅出 GAN·原理篇文字版（完整）](https://mp.weixin.qq.com/s/dVDDMXS6RA_NWc4EpLQJdw)


### 待分类




### GAN Theory


### 高质量图片生成

#### DCGAN

- []Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[[Paper]](https://arxiv.org/abs/1511.06434)[[Tensorflow--code]](https://github.com/carpedm20/DCGAN-tensorflow)(ICLR 2016)

其他代码：

- [dcgan-completion.tensorflow](https://github.com/bamos/dcgan-completion.tensorflow)
- [keras-dcgan](https://github.com/jacobgil/keras-dcgan)

学习笔记：

- [x] 2018-12-10:  [[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(上）](https://mp.weixin.qq.com/s/S_uiSe74Ti6N_u4Y5Fd6Fw)
- [x] 2018-12-16：[[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(中）](https://mp.weixin.qq.com/s/nYDZA75JcfsADYyNdXjmJQ)
- [x] 2018-12-22：[[GAN学习系列3]采用深度学习和 TensorFlow 实现图片修复(下）](https://mp.weixin.qq.com/s/1Q39H4bA_2k3e4ry5zSQZQ)

相关文章：

- [Notes for DCGAN paper](https://gist.github.com/shagunsodhani/aa79796c70565e3761e86d0f932a3de5)
- [想实现 DCGAN？从制作一张门票谈起！](https://www.jiqizhixin.com/articles/2018-01-16-4)


### Super-Resolution

- (ECCV 2018)ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks[[paper]](https://arxiv.org/abs/1809.00219)[[code]](https://github.com/xinntao/ESRGAN)


相关介绍文章：

- [效果惊人：上古卷轴III等经典游戏也能使用超分辨率GAN重制了](https://mp.weixin.qq.com/s/eJkkbGBYxWlngkT5gjjW7g)
- [让画面更逼真！这个强化超分辨率GAN让老游戏迎来第二春 | 代码+论文+游戏MOD](https://mp.weixin.qq.com/s/kobEEizpP2v5Yy-8stiGgg)
- [如何在Windows上运行ESRGAN](https://kingdomakrillic.tumblr.com/post/178254875891/i-figured-out-how-to-get-esrgan-and-sftgan)


### Inpainting

- [ ] Generative Image Inpainting with Contextual Attention[[Paper]](https://arxiv.org/abs/1801.07892)[[code]](https://github.com/JiahuiYu/generative_inpainting)
- [ ] EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning[[Paper]](https://arxiv.org/abs/1901.00212)[[Code]](https://github.com/knazeri/edge-connect)[[介绍]](https://mp.weixin.qq.com/s/F8o_zBBvuWyW90uyP5bLvQ)

#### Project

- [动漫人物图片自动修复，去马赛克，填补，去瑕疵](https://github.com/youyuge34/Anime-InPainting)--基于 EdgeConnect 这篇论文的基础
- Painting Outside the Box: Image Outpainting[[Paper]](https://cs230.stanford.edu/projects_spring_2018/posters/8265861.pdf)[[Code]](https://github.com/bendangnuksung/Image-OutPainting)[[介绍]](https://zhuanlan.zhihu.com/p/40902853)--吴恩达斯坦福 CS230 课程期末作业第一名：图像超级补全效果惊艳


### 工具

- [x] 谷歌开源的 TFGAN 库[[original article]](Generative Adversarial Networks: Google open sources TensorFlow-GAN (TFGAN))[[Github]](https://github.com/tensorflow/models/tree/master/research/gan)
- [x] 基于 Pytorch 的 TorchGAN[[Github]](https://github.com/torchgan/torchgan)

### 综合文章

#### Github

1. 复现多种 GANs 模型[[Tensorflow-version]](https://github.com/hwalsuklee/tensorflow-generative-model-collections)[[Pytorch]](https://github.com/eriklindernoren/PyTorch-GAN)[[Keras]](https://github.com/eriklindernoren/Keras-GAN)
2. [tensorflow-GANs](https://github.com/TwistedW/tensorflow-GANs)--结合了多种现在的GAN代码，部分有中文注释的Github
3. [AdversarialNetsPapers](https://github.com/zhangqianhui/AdversarialNetsPapers)--及时更新，而且对论文进行分类
4. [The-gan-zoo](https://github.com/hindupuravinash/the-gan-zoo)


#### 文章

1. [干货 | 深入浅出 GAN·原理篇文字版（完整）](https://mp.weixin.qq.com/s/dVDDMXS6RA_NWc4EpLQJdw)
2. [微软剑桥研究院153页最新GAN教程](https://mp.weixin.qq.com/s/zHyB3Hor7OrvTKkLN_M7_Q)--github:https://github.com/nowozin/mlss2018-madrid-gan


