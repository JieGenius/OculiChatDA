# OculiChatDA
眼科问诊大模型

## 环境安装
```bash
conda create -n OculiChatDA python=3.10 # 不建议安装3.11以及以上版本, xtuner最新版只支持3.8~3.10
conda activate OculiChatDA
pip install -r requirements.txt
```

## 数据集
### 通用语料数据集
来自书生万卷1.0， 地址如下:
[数据集地址](https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0/tree/main/raw/nlp/CN/ChinaNews-cn)  part-006853-a894b46e.jsonl.tar.gz
下载到 data/raw_data目录下

### MSAgent数据集
确保模型不会遗忘Agent能力 (暂未添加)

### 问诊数据集
1. data/processed_data/ophthalmology_9th Edition.json
2. data/processed_data/ophthalmology_qa.json
3. data/processed_data/中医眼科.json
4. data/processed_data/MedDialog/*

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

# 开始训练
xtuner train config/internlm2_chat_7b_qlora_med_dialog_e5_copy.py --deepspeed deepspeed_zero2 
# 实测batch为4耗显存26G，需要开一个2 * 1/4的机器

```

