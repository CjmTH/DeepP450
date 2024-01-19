# ***DeepP450: Predicting Human P450 Activities of Small Molecules by Integrating Pretrained Protein Language Model and Molecular Representation***


![model](https://github.com/CjmTH/DeepP450/assets/156410487/5064bb10-a6e8-46cf-9a00-cadb580ce710)



## 配置环境
 ### Uni-mol
   * Uni-Mol 依托于深势科技开发的高性能分布式框架 Uni-Core，因此，应该先安装 Uni-Core, 可以直接参照 Uni-Core 的官方代码仓库，下面提供一种可能的配置方案。

      比如CUDA 版本为11.3，则可以使用如下命令：
   
      pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

   * 下载 Uni-Mol 的代码，进行安装 [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)
   
      cd Uni-Mol/unimol
   
      pip install

   * 下载Uni-Mol预训练模型权重，放在main文件夹。
    https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt

 ### ESM-2
   * 本模型中ESM-2为esm2_t33_650M_UR50D版本，需要先下载ESM-2权重文件并放在main文件文件夹。 https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

      此外，需要对ESM-2的源码pretrained.py 进行以下改变：
   
 	  ![微信图片_20240117141147](https://github.com/CjmTH/DeepP450/assets/156410487/17a9b67a-3b06-449f-a2e3-e114f8979469)

   * 我们基于真核生物P450序列对ESM-2进行了微调，下载微调模型权重后放在main文件夹。

   



## 模型训练与测试

1. 数据格式可根据data/data_0/raw/train_0.csv格式进行编辑，data文件夹下需首先创建raw、intermediata和result三个文件夹，用于分子处理过程中文件生成；

2. 本模型包含20个子模型，下载子模型权重后放在weight文件夹下，运行model.py即可； https://www.alipan.com/s/Wyy14DP8MNA

3. 最终预测结果整理为total_soft.csv文件后运行metric.py即可获得最终集成模型软投票结果。 
