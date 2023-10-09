1、run_train.sh
    使用单机多卡方式并行训练
2、kill_proc.sh
    删除训练过程中异常终止的僵尸进程

3、train.py
    训练文件，由run_train.sh调用
4、predict_begin.py
    推理启动文件，读取数据，初始化网络
5、predict.py
    推理文件，数据预处理，模型推理，结果比对

6、nets/modules.py
    Siamese网络结果，使用vgg16的backbone提取特征后，
    将两张图片的特征值做L1距离，输出预测值。
7、nets/vgg.py
    vgg16基础网络，用于提取特征

8、utils/dataset.py
    读取数据集，并将输入数据准备成pairs形式
9、utils/utils_aug.py
    数据增强
10、utils/utils_fit.py
    由train.py调用，遍历batch数据，进行前向传播和方向传播
11、utils/utils.py
    其他相关工具

12、 log
    pth --- 模型文件
    log --- 日志文件
    loss_2023_1-_07_12_02_52 --- tensorbord格式训练文件

13、datasets 数据集目录
