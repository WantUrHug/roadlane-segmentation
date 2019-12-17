# roadlane-segmentation
用语义分割的方式来做车道线检测 <br>
原本的想法是使用语义分割的方式，来识别经过逆透视变换之后的俯视图中的车道线，参考了[这篇论文](https://arxiv.org/abs/1812.05914)中提到的模型。论文中是是多分类，对于我的需求只需要划分车道线和背景即可，所以输出的通道数为1，之后通过设置阈值可以提高测试图片测试结果的表现。
![原始图片](https://github.com/WantUrHug/roadlane-segmentation/blob/master/images/origin.png "原始图片")
![模型输出](https://github.com/WantUrHug/roadlane-segmentation/blob/master/images/threshold_0.5.png "阈值为0.5")
![模型输出](https://github.com/WantUrHug/roadlane-segmentation/blob/master/images/threshold_0.7.png "阈值为0.7")
![模型输出](https://github.com/WantUrHug/roadlane-segmentation/blob/master/images/threshold_0.9.png "阈值为0.9")
