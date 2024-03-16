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
    <img src="assets/logo.png" alt="Logo" width="60%">
  </a>

<h3 align="center">OculiChatDA</h3>
  <p align="center">
    <br />
    <a href="https://openxlab.org.cn/apps/detail/flyer/oculi_chat_diagnosis_assistant"> OpenXLab 体验</a>
    ·
    <a href="https://github.com/JieGenius/OculiChatDA/issues">报告Bug & 提出新特性</a>
  </p>
</p>

🎉更新

- \[2024/03/15\] 将病种数量提升至4，模型训练时做了数据清洗并使用了msagent的部分数据以提高工具调用能力
- \[2024/02/20\] Lagent版本支持0.2.2

## 简介

</br>

**OculiChatDA** 是一个眼科问诊的大模型，用户可以对眼睛相关的疾病进行问诊，获取专业的医学建议。同时模型拥有读图的能力，可通过眼底图判断是否为**青光眼或糖尿病视网膜病变**患者。
OculiChatDA的问诊数据集包含了眼科问诊的常见问题，可以进行多轮对话，支持多种对话场景，包括问诊、咨询、闲聊等。

开源不易，如果本项目帮到大家，可以右上角帮我点个 star~ ⭐⭐ , 您的 star ⭐是我们最大的鼓励，谢谢各位！

## 介绍

据估算，中国的医生和患者比例仅为西方国家的1/6。尽管国家在医疗卫生事业投入不断增加，医院数量和规模也大幅提升，但医生数量和质量仍然存在瓶颈。与此同时，城乡医疗资源的不均衡分布给农村和偏远地区的居民带来了极大的就医不便。

本项目基于以上现状，设计了一款低成本，先进的对话问诊系统，通过对话的方式，帮助患者迅速进行初步的诊断，并且给出合理的建议，同时，该模型已经初步具有“望闻问切”的一半能力，即**望**和**问**。这一低成本、高效率的解决方案致力于填补医疗资源不足的空缺，为更多患者提供及时、便捷的医疗服务。

## 演示

**Demo访问地址:** [OculiChatDA](https://openxlab.org.cn/apps/detail/flyer/oculi_chat_diagnosis_assistant)

**效果展示:**
![宣传图](assets/publicity.png)

## 模型

| 模型                  | 基座                | 数据量    | OpenXLab
|---------------------|-------------------|--------|----------------------------------------------------------------------------------------|
| OculiChatDA-chat-7b | InternLM2-chat-7b | 60K个对话 | [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/flyer/OculiChatDA/tree/v1) |

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
      "system": "你是一名眼科专家，你需要解答患者的疑问，提供准确的回答，必要时，提醒患者及时挂号就医。\n当患者对症状描述不清时，你需要循序渐进的引导患者，询问患者的症状，以便给出准确的诊断。\n当患者上传眼底图时，你需调用Agent判断眼底图中是否存在青光眼和糖尿病视网膜病变，若存在，需要提醒患者及时就医。\n",
      "input": "什么是单纯疤疹病毒性角膜炎？",
      "output": "单纯疤疹病毒性角膜炎是由单纯疤疹病毒（HSV）引起的角膜感染，是一种常见的眼部疾病，也是致盲性角膜病的主要原因之一。它的特点是反复发作，多次发作会导致角膜逐渐混浊并最终可能导致失明。"
    }
  ]
}

```

## 训练细节

见 [details.md](details.md)

## Web Demo

```bash
streamlit run web_demo.py --server.address=0.0.0.0 --server.port 7860 --server.enableStaticServing True
```

## 眼底病变诊断模型

目前支持青光眼分类、糖尿病视网膜病变分级、病理性近视分类，年龄相关性黄斑变性分类。

Action代码 [fundus_diagnosis.py](utils/actions/fundus_diagnosis.py)

模型训练代码详见： [JieGenius/GlauClsDRGrading](https://github.com/JieGenius/GlauClsDRGrading/tree/amd_pm)

## 项目参与人员

1. 杨亚杰，组长，中国人民大学硕士毕业，现在某公司做算法工程师
2. 李爽，职业算法工程师，主要从事于医疗算法的研发与研究。
3. 江天松，南京理工大学计算机科学与工程学院硕士研究生。
4. 李俊杰，中国人民大学研究生毕业，主要从事数据挖掘、数据分析建模相关工作。
5. 徐翼萌，本科毕业于山东大学（威海），目前在中科院深圳先进技术研究院合成所做科研助理。

## 特别感谢

感谢上海人工智能实验室组织的**书生·浦语实战营**学习活动～～～

感谢[OpenXLab](https://openxlab.org.cn/)提供的算力支持

感谢浦语小助手对项目的支持

本项目的README部分大量参考了[TheGodOfCookery](https://github.com/zhanghui-china/TheGodOfCookery)～～～

## TODO

- [ ] 工具调用能力微调。 （目前通过打了多个补丁，临时解决重复调用，参数错误调用的问题，）
- [ ] 语音问诊
- [ ] 视频问诊，数字人接入。
- [ ] 问诊数据集扩充，增加更多真实的问诊数据
- [ ] Agent能力扩充，识别更多的眼病（如中心性浆液，病理性近视，视网膜脱离等），更多的模态（如OCT，裂隙灯，眼表照相等），

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JieGenius/OculiChatDA&type=Date)](https://star-history.com/#JieGenius/OculiChatDA&Date)
