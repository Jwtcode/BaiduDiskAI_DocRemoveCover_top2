# 百度网盘AI大赛-文档图片去遮挡比赛第2名方案


# 一、赛题分析
此次大赛主题结合日常生活常见情景展开，人们在使用手机等移动设备扫描证件或者扫描文档、拍摄展示资料的场景中，经常会拍摄到一些手指或者人头等其他因素，对扫描成品的美观和易用性产生了影响。期望同学们通过计算机技术对给定文档图像进行处理，帮助人们去除文档图像中的手指、人头等因素，还原真实的文档资料，提升使用效率。

# 二、 数据分析
- 本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中训练集共14400组样本，A榜测试集共320个样本，B榜测试集共640个样本,抽取一部分数据如图：
![image](https://github.com/Jwtcode/BaiduDiskAI_DocRemoveCover_top3/blob/master/illustration/0_IMG_20220705_103123_a.jpg)
![image](https://github.com/Jwtcode/BaiduDiskAI_DocRemoveCover_top3/blob/master/illustration/0_IMG_20220705_103123_a_0_000.jpg)
![image](https://github.com/Jwtcode/BaiduDiskAI_DocRemoveCover_top3/blob/master/illustration/0_IMG_20220705_103123_a_0_000.png)
- hand,head 为带有手指、人头遮挡数据及遮挡部位分割图，gt 为非遮挡图片（仅有训练集数据提供gt ，A榜测试集、B榜测试集数据均不提供gt);
- annotation.txt 为图片对应关系，每一行为一组样本，用空格分隔，分别为非遮挡图片、遮挡图片、分割图片;
# 三、模型设计
针对图像去遮挡这个任务，我们查阅了相关资料，认为该任务是一个先擦出后重建优化的过程，所以我们选择了目前常用的EraseNet作为我们此次的baseline。整体思路分为预测目标所在位置，裁剪目标所在区域，初步去遮挡，去遮挡优化，生成mask，替换mask所在区域。
- 预测目标所在位置：<br>
测试图片直接缩放成640x640进行预测并获得boundingbox。<br>
![image](https://github.com/Jwtcode/BaiduDiskAI_DocRemoveCover_top3/blob/master/illustration/575.jpg)

- 裁剪目标所在区域<br>
这里分为以下几种情况：<br>
1）预测的boundingbox的宽高都小于1280<br>
目标区域向四周扩充为1280x1280。<br>
2）预测的boundingbox的宽高有一个大于或者全都大于1280<br>
大于1280的边长向外扩充为64的倍数即可（避免出现维度不匹配的情况出现）。<br>
3）预测出多个boundingbox<br>
将多个boundingbox按面积由大到小排序，面积最大的boundingbox最先扩充，如果后面的boundingbox包含在这个扩充的boundingbox里面，则不进行处理，如果不包含其中，将其扩充为同样的尺寸，保证输入到模型中的宽高一致。
- 初步去遮挡:改进后的EraseNet去除了预测mask的分支（训练中发现预测mask和去遮挡有冲突），整体可以看成unet结构。
- 去遮挡优化网络<br>
去遮挡优化网络采用U型结构的自监督学习的idr网络，通过再次的编解码进行去遮挡的二次优化，得到最终的去遮挡图。
- 生成mask<br>
利用yolox预测的boundingbox(不扩充)在优化后的去遮挡图上裁剪出目标区域，将裁下的图片与输入图片目标所在区域做差，差值的绝对值大于等于2的地方作为mask。
- 替换mask所在区域<br>
将mask在原图中的位置的像素替换为mask在去遮挡图位置的像素。<br>

- 网络主体架构为UNet，如图：
![image](https://github.com/Jwtcode/BaiduDiskAI_DocRemoveCover_top3/blob/master/illustration/pipeline.png)
从网络结构图上可以直观的看出改进后EraseNet变成了单分支网络，这是因为原版EraseNet的预测mask分支和第一阶段的Decoder存在冲突，所以我们去掉了预测mask分支，考虑到实效性，我们没有额外训练一个分割模型，而是选择检测模型(yolox)来获得mask。在损失函数上，原版的ErastNet使用了感知损失以及GAN损失，这个损失函数是为了生成更加逼真的背景，但是本赛题任务的背景都是纯色，所以这两个损失是不需要的。此外，EraseNet在多个尺度上使用了l1损失，我们只在第一阶段和第二阶段的最后一个尺度上使用了l1损失。此外，根据经验，我们将EraseNet的重建网络Refinement替换为了idr网络并在底层叠加了non-local结构。


# 四、数据处理与增强

### 数据划分
- yolox<br>
官方给的数据为14400张图片，所有图片直接缩放为640x640，其中12000张用于训练，2400张用于验证。
利用数据集自带的mask生成yolo格式标签,选用yolox作为检测模型。
- EraseNet<br>
官方给的数据为14400张图像，我们裁剪出包含遮挡物的图片为10800张（全部为正样本，尺寸为1024x1024），其中9000张用作训练，1800张用于验证。
### 数据增广
- 增强使用横向左右翻转和小角度旋转

# 五、训练细节
- 训练配置
总迭代数：450000 iteration<br>
我们采用batch size为4和patch size为1024来进行训练450000次迭代。<br>
我们采用了余弦退火的学习率策略来优化网络，学习率我们采用1e-4，优化器为Adam。<br>
- 损失函数为L1Loss

# 六、测试细节
- 原图缩放为640x640输入到yolox网络中获得boundingbox，除以缩放比例获得在原图上的boundingbox，将boundingbox所在区域扩充得到去遮挡网络的输入。
- 裁剪的图片输入到去遮挡网络中获得输出。
- 输出与输入做差获得mask.
- 将原图mask所在区域的像素值替换为输出mask所在区域的像素值。
