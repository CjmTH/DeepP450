# ***DeepP450: Predicting Human P450 Activities of Small Molecules by Integrating Pretrained Protein Language Model and Molecular Representation***


![model](https://github.com/CjmTH/DeepP450/assets/156410487/5064bb10-a6e8-46cf-9a00-cadb580ce710)



## Setting up
 ### Uni-mol
   * Uni-Mol is built upon the high-performance distributed framework Uni-Core, developed by DeepTech. Consequently, it is advisable to prioritize the installation of Uni-Core. This can be accomplished by referring directly to the official code repository of Uni-Core. Below, I provide a potential configuration scheme for this process.

      For example, if the CUDA version is 11.3, you can use the following command:
   
      pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

   * Download the code Uni-Mol code and install. [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)
   
      cd Uni-Mol/unimol
   
      pip install

   * Download the Uni-Mol weight file and put in the main folder。
    https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt

 ### ESM-2
   * In this model, ESM-2 refers to the esm2_t33_650M_UR50D version. Please download the ESM-2 weight file and place it in the main folder. https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

      Additionally, modifications need to be made to the source code file pretrained.py for ESM-2 as follows:
   
 	  ![微信图片_20240117141147](https://github.com/CjmTH/DeepP450/assets/156410487/17a9b67a-3b06-449f-a2e3-e114f8979469)

   * We fine-tuned ESM-2 model using eukaryotic P450 sequences. After downloading the fine-tuned model weight, place this file in the main folder.

   



## model training and prediction

1. The input file format can be edited according to data/data_0/raw/train_0.csv. Within the data folder, create three subfolders named raw, intermediate, and result respectively. These folders are utilized for the generation of files during the molecular processing process；

2. The model comprises 20 sub-models. After downloading the weight files for these sub-models, place them in the weight folder. Then, execute model.py to commence the process. Besides, the ESM-2 fintuned weight can also be accessed. （ https://pan.baidu.com/s/1riqf9yRcIbjbtNS5PjQzag access code: uqxt）

   
4. Final results are organized into a total_soft.csv file. running metric.py will yield the final ensemble model's soft voting results.
