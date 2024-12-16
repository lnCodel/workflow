
# 抑郁症检测工作流

## 简介
本工作流专注于抑郁症检测任务，基于图神经网络方法，能够快速、精准地完成组织分类。该工作流适合应用于复杂病理场景中，为病理诊断和研究提供技术支持。

### 核心特性
本工作流基于图神经网络实现抑郁症检测。首先，需要使用DPARSF对原始图像进行预处理。之后，根据用户输入的数据再次进行预处理得到每个患者的特征，根据患者特征进行构图。图中一个节点表示一个患者，图中的边表示患者之间的关系，由模型计算得出，经过模型推理可以得出图中每个节点是否患有抑郁症的概率，结果存于results文件下。结果包含三种：1）构图可视化；2）经过模型推理得到的新的图可视化；3）患者抑郁症检测概率。
所推理的模型在私有数据集中的五折交叉验证的结果上，acc可以达到80%以上，auc达到了75%以上，满足任务书中的指标要求。
- **适用场景**：大规模病理图像分类分析和临床辅助诊断。

---

## 环境要求
### 硬件环境
- **CPU**: Intel(R) Xeon(R) Gold 6230 @ 2.10GHz  
- **内存**: 256GB  
- **GPU**: NVIDIA GeForce GTX 3090  
- **存储**: 1TB HDD  

### 软件环境
- **Python**: 3.7  
- **Pytorch**: 1.7.1  
- **Torchvision**: 0.8.2  
- **Nextflow**: 24.10.1  

---

## 数据说明
### 输入数据
- MRI数据需要提前进行预处理，预处理的步骤参考提交的工作流文档。
- 在提取完影像组学特征后，其中fMRI数据需要放置到./depression3/test/fMRI/，进入该路径下，根据不同的脑图谱将数据放置到对应的脑图谱路径下。
-其中sMRI数据需要放置到./depression3/test/sMRI/，进入该路径下，根据不同的脑图谱将数据放置到对应的脑图谱路径下。


---

## 快速开始
### 环境安装（如果直接有安装pytorch的环境，可以忽略这一步）
1. 创建并激活Python虚拟环境：
   ```bash
   conda create -n DARC python=3.7
   conda activate DARC
   ```
2. 安装依赖：
   ```bash
   pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
   pip install -r requirements.txt
   ```

### 运行流程
 将需要推理的数据放到depression3文件夹下。
其中fMRI数据需要放置到./depression3/test/fMRI/，进入该路径下，根据不同的脑图谱将数据放置到对应的脑图谱路径下。
其中sMRI数据需要放置到./depression3/test/sMRI/，进入该路径下，根据不同的脑图谱将数据放置到对应的脑图谱路径下。
可选的脑图谱有aal_90; aal_116; bn_246; CC200; CC400;	
3. **查看结果**：分类结果存储在`results`文件夹中，执行以下命令即可查看：

---

## 测试数据集 
- **数据规模**：71个受试者。  
- **分类类别**：是否患有抑郁症。
- **实验结果**：五折交叉验证的分类准确率80%以上。

---
