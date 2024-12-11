# YOLOV5_for_PyTorch_v7.0

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)



# 概述

## 简述

YOLO是一个经典的物体检测网络，将物体检测作为回归问题求解。YOLO训练和推理均是在一个单独网络中进行。
基于一个单独的end-to-end网络，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov5/tree/v7.0
  commit_id=a9f895d304aea5920e694606927fa9208aa7f0ed
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/modelZoo-GPL.git
  code_path=built-in/PyTorch/Official/cv/object_detection
  ```

- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

- 通过单击“立即下载”，下载源码包。



# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套       | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 硬件       | [1.0.17](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | 固件与驱动  | [6.0.RC1](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.0.RC1](https://www.hiascend.com/software/cann/commercial?version=6.0.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)       |


- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  ```
  pip3.7 install -r requirements.txt
  ```
- 编译安装torchvision

  ***为了更快的推理性能，请编译安装而非直接安装torchvision***

   ```
    git clone -b v0.9.1 https://github.com/pytorch/vision.git #根据torch版本选择不同分支
    cd vision
    python setup.py bdist_wheel
    pip3 install dist/*.whl
   ```

- 编译安装Opencv-python。

   为了获得最好的图像处理性能，请编译安装opencv-python而非直接安装。编译安装步骤如下：

   ```
    export GIT_SSL_NO_VERIFY=true
    git clone https://github.com/opencv/opencv.git
    cd opencv
    mkdir -p build
    cd build
    cmake -D BUILD_opencv_python3=yes -D BUILD_opencv_python2=no -D PYTHON3_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m -D PYTHON3_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m -D PYTHON3_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/python3.7.5/lib/python3.7/site-packages/numpy/core/include -D PYTHON3_PACKAGES_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages -D PYTHON3_DEFAULT_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m ..
    make -j$nproc
    make install
   ```
  
## 准备数据集


   用户自行获取coco数据集，包含images图片和annotations文件。其中images图片和annotations文件从[coco官网](https://cocodataset.org/#download)获取，另外还需要labels图片，用户可以从[google drive](https://drive.google.com/uc?export=download&id=1cXZR_ckHki6nddOmcysCuuJFM--T-Q6L)中获取。将获取后的数据集解压放置服务器的任意目录下(建议放到源码包根目录XXX/coco/下)。

  数据集目录结构如下所示：

   ```
   coco
       |-- annotations
       |-- images
          |-- train2017
          |-- val2017   
       |-- labels
          |-- train2017
          |-- val2017
   ```

## 数据集预处理

   生成yolov5专用标注文件:

   1. 将代码仓中cocofile/coco2yolo.py和cocofile/coco_class.txt拷贝到coco数据集**根目录**。
   2. 运行coco2yolo.py。
       ```
       python3 coco2yolo.py
       ```
   3. 运行上述脚本后，将在coco数据集**根目录**生成train2017.txt和val2017.txt。
   4. 建立数据集软连接：
        ```
        cd Yolov5_for_PyTorch_v7.0
        ln -s /data/to/coco coco
        ```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd /${模型文件夹名称}
   ```

2. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

   - 单机单卡训练

     启动单卡训练。

     ```
     bash ./test/train_yolov5s_full_1p.sh   # 1p精度    
     bash ./test/train_yolov5s_performance_1p.sh   # 1p性能
     ```

   - 单机8卡训练

     启动8卡训练。

     ```
     bash ./test/train_yolov5s_full_8p.sh   # 8p精度    
     bash ./test/train_yolov5s_performance_8p.sh   # 8p性能

     ```
   - 多机多卡训练指令
   
     启动多机多卡训练。
     ```
     bash test/train_yolov7_cluster.sh --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
     ```
     ps:脚本默认为8卡，若使用自定义卡数，继续在上面命令后添加 --device_number=每台机器使用卡数 --head_rank=起始卡号，例如分别为4、0时，代表使用0-3卡训练。

     --epochs传入训练周期数，默认300， --batch_size传入模型total batch size，可以以单卡batch_size=64做参考设置。

     
   - 模型评估。

       ```
       bash ./test/train_yolov5s_eval_1p.sh 
       ```

   模型训练脚本参数说明如下。

      ```
      公共参数：
      --device                            //训练指定训练用卡
      --img-size                          //图像大小
      --data                              //训练所需的yaml文件
      --cfg                               //训练过程中涉及的参数配置文件
      --weights                           //权重
      --batch-size                        //训练批次大小
      --epochs                            //重复训练次数，默认：300
      ```
   
      训练完成后，权重文件保存在当前路径下，并输出模型训练精度和性能信息。


# 训练结果展示

**表 2**  训练结果展示表

| NAME     | mAP    | FPS    | Epochs | Torch_version |
|--------  | ------ |:-------| ------ | :------------ |
| 1p-竞品  | -      | 181    | 1      | -             |
| 8p-竞品  | 0.347  | 1264   | 300    | -             |
| 1p-NPU   | -      | 196.1  | 1      | 1.8           |
| 8p-NPU   | 0.350  | 1482.1 | 300    | 1.8           |

# 版本说明

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

## 变更

2023.01.12：更新Readme发布。

## 已知问题

无。
