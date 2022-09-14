# FlowerShop_wbl
#项目已开发完毕，最终版已打包
整个系统下载链接：链接：https://pan.baidu.com/s/1rEvVGCtxJSRDpBKE0HCKgQ 
提取码：1111 
--来自百度网盘超级会员V6的分享

这个项目是之前疫情做的一个识别项目，稍加修改成为了电商网站。以下是我的文档描述：
本项目主要为一个基于花卉商城和web实现的花卉识别网站。图像识别的实践和与电商网站的结合，是本次毕业设计任务的出发点。程序设计分为两个主要部分，一是实现网站部分，二是实现花卉识别的图像识别技术TensorFlow部分。根据本科毕业设计项目的内容，本文系统地描述了花卉识别与销售网站的一般功能和实现。同时设计并实现了花卉识别样例的寻找，花卉识别模型的构建，上传花卉照片，识别花卉，将其添加到商品管理、数据库管理、原材料分类和查询、购物车和订单定制处理中。当赏花者看到自己喜欢的花时，他们可以拍照、获得花卉的名字并立即购物。
技术语言：
# MySQL：它对数据处理速度快、同时它作为一个开源的软件，支持很多种PC平台，提供的接口支持Java操作等等。对于编程，两个基本类（其中一种为JDBC）可以连接到数据库。
# Java：它是一种不受计算机和系统平台制约的编程语言，它可以随处运行，非常适合于网络编程。在今天，Java也是榜首语言。
# Python：主要是作为TensorFlow的模型构建编程语言，Python比较先进，速度快接口丰富，同时完美对接tf.js，总的来看在本项目用于图像识别的开发语言。
# TensorFlow2.0：选用TensorFlow2.0作为生成花卉模型的工具，没有选择pytorch是因为tf2的keras和pytorch构建训练图的代码已经相差无几了，因为一直学的都是TensorFlow，没必要再去接触另一种语言。机器学习和深度学习的所有前向传播的过程，以及数据处理、信号处理的过程，都可以抽象看做是对数据进行操作（加减乘除）得到的结果。TensorFlow的设计易读且效率高，将数据保存在张量里，既保证了开发算法时使用python接口的便利性，也保证了部署模型的运行效率，采用6年前的GTX1080Ti也可获得最棒的体验（跑样例和模型运算速度），同时保证了经济可行性。最重要的是他可以与js无缝衔接。
# SSM：它作为关键技术不仅在各种工作场合频繁用到，也是现在轻量级方案中，传播的最为广泛的架构。因此采用此架构。
# Tomcat：其作为web服务器。被视为Apache扩展。如果查询是静态网站，谷歌会处理并返回结果。对于动态查询，会在处理后返回结果，不仅可提高性能，也可以做到负载平衡。

