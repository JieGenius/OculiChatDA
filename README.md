# OculiChatDA 眼科问诊大模型
<!-- PROJECT SHIELDS -->
<!-- 
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Stargazers][stars-shield]][stars-url]
-->

<br />
<!-- PROJECT LOGO -->

<p align="center">
  <a href="https://github.com/JieGenius/OculiChatDA/">
    <img src="assets/logo.png" alt="Logo" width="30%">
  </a>

<h3 align="center">OculiChatDA</h3>
  <p align="center">
    <br />
    <a href="/"> OpenXLab 体验</a>
    ·
    <a href="https://github.com/JieGenius/OculiChatDA/issues">报告Bug & 提出新特性</a>
  </p>
</p>

## 简介

</br>

**OculiChatDA** 是一个眼科问诊的大模型，用户可以对眼睛相关的疾病进行问诊，获取专业的医学建议。同时模型拥有读图的能力，可通过眼底图判断是否为**青光眼或糖尿病视网膜病变**患者。
OculiChatDA的问诊数据集包含了眼科问诊的常见问题，可以进行多轮对话，支持多种对话场景，包括问诊、咨询、闲聊等。 模型用xtuner在InternLM2的基础上指令微调而来，并使用Lagnet0.1扩展了模型的读图能力。

开源不易，如果本项目帮到大家，可以右上角帮我点个 star~ ⭐⭐ , 您的 star ⭐是我们最大的鼓励，谢谢各位！

## 介绍
据估算，中国的医生和患者比例仅为西方国家的1/6。尽管国家在医疗卫生事业投入不断增加，医院数量和规模也大幅提升，但医生数量和质量仍然存在瓶颈。与此同时，城乡医疗资源的不均衡分布给农村和偏远地区的居民带来了极大的就医不便。

本项目基于以上现状，设计了一款低成本，先进的对话问诊系统，通过对话的方式，帮助患者迅速进行初步的诊断，并且给出合理的建议，同时，该模型已经初步具有“望闻问切”的一半能力，即**望**和**问**。这一低成本、高效率的解决方案致力于填补医疗资源不足的空缺，为更多患者提供及时、便捷的医疗服务。

## 演示

Demo访问地址: [OculiChatDA](http://)

## 模型
| 模型                  | 基座                | 数据量               | OpenXLab ｜                                                                             
|---------------------|-------------------|-------------------|----------------------------------------------------------------------------------------|
| OculiChatDA-chat-7b | InternLM2-chat-7b | 500K个对话，对话轮次为1～64 | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)]() |

## 环境安装
```bash
conda create -n OculiChatDA python=3.10 # 不建议安装3.11以及以上版本, xtuner最新版只支持3.8~3.10
conda activate OculiChatDA
pip install -r requirements.txt
```

## 数据集
### 问诊数据集

1. 眼科专业书籍
2. 眼科习题
3. 中医眼科
4. [MedDialog数据集](https://github.com/UCSD-AI4H/Medical-Dialogue-System)

样例如下:
```json
{
  "conversation": [
    {
      "system": "你是一名眼科专家，你需要解答患者的疑问，提供准确的回答，必要时，提醒患>者及时挂号就医。\n当患者对症状描述不清时，你需要循序渐进的引导患者，询问患者的症状，以便给出准确的诊断。\n当患者上传眼底图时，你需调用Agent判断眼底图中是否存在青光眼和糖尿病视网膜病变，若存在，需>要提醒患者及时就医。\n",
      "input": "什么是单纯疤疹病毒性角膜炎？",
      "output": "单纯疤疹病毒性角膜炎是由单纯疤疹病毒（HSV）引起的角膜感染，是一种常见的眼部疾病，也是致盲性角膜病的主要原因之一。它的特点是反复发作，多次发作会导致角膜逐渐混浊并最终可能导致失明。"
    }
  ]
}

```
## 微调
```bash
mkdir -p /root/OculiChatDA/data
ln -s /share/model_repos/internlm2-chat-7b /root/OculiChatDA/data
xtuer list-cfg
xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 config
cd config
mv internlm2_chat_7b_qlora_oasst1_e3_copy.py  internlm2_chat_7b_qlora_med_dialog_e5_copy.py
vim internlm2_chat_7b_qlora_med_dialog_e5.py # 修改配置文件
---> pretrained_model_name_or_path=/data/interlm2-chat-7b
---> max_epochs = 5
---> data_path = "./data/qa_data.json"
---> batch_size = 4
---> evaluation_inputs = evaluation_inputs = ['青光眼诊断的三要素是什么？', '糖尿病和糖尿病视网膜病变有什么关系呢', "医生你好，我的视野中心有黑色阴影，这是为什么呢?"]
---> dataset=dict(type=load_dataset, path="json", data_files=dict(train=data_path)),
---> dataset_map_fn = None

## Web Demo
```bash
streamlit run web_demo.py --server.address=0.0.0.0 --server.port 7860
```
# 训练
```bash
xtuner train config/internlm2_chat_7b_qlora_med_dialog_e5_copy.py --deepspeed deepspeed_zero2 
# 实测batch为4耗显存26G，需要开一个2 * 1/4的机器
```

## TODO
- [ ] 问诊数据集扩充，增加更多真实的问诊数据
- [ ] Agent能力扩充，识别更多的眼病（如中心性浆液，病理性近视，视网膜脱离等），更多的模态（如如OCT，裂隙灯，眼表照相等），