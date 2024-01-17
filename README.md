配置环境
1. # Uni-mol
   Uni-Mol 依托于深势科技基于 pytorch 开发的高性能分布式框架 Uni-Core，因此，应该先安装 Uni-Core, 可以直接参照 Uni-Core 的官方代码仓库，下面提供一种可能的配置方案。

   比如CUDA 版本为11.3，则可以使用如下命令：
   
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

   下载 Uni-Mol 的代码，进行安装 [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)
   
   cd Uni-Mol/unimol
   
   pip install

3. # ESM-2
   我们选择的是esm2_t33_650M_UR50D参数的ESM-2模型，需要先下载ESM-2权重文件与main.py文件放在同一个文件夹（https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt）

   此外，需要对ESM-2的源码pretrained.py 进行以下改变：
   
    ![微信图片_20240117141147](https://github.com/CjmTH/DeepP450/assets/156410487/17a9b67a-3b06-449f-a2e3-e114f8979469)



运行模型训练与测试

1. 数据格式可根据data/data_0/raw/train_0.csv格式进行编辑，data文件夹下需首先创建raw、intermediata和result三个文件夹，用于分子处理过程中文件生成。
   
3. 运行main.py即可。
