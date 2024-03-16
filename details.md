# 训练微调细节

## 数据清洗说明

对于MedDialog数据集， 我们首先通过关键字筛选和眼科相关的对话数据，然后使用 "InternLM2-chat-20b-4bits" 大模型来判断数据质量，与眼科的相关度。（仅保留了1/6的数据）
其他数据集，在我们最新的训练中并未使用。我们会在将来通过RAG的方式嵌入到我们的应用中，不再将其用于微调大模型。

为了保持大模型工具调用的能力，我们也添加了一些msagent的数据，
相关代码见 [tools/data_prepare/data_clean.py](tools/data_prepare/data_clean.py) 以及 [tools/data_prepare/merge_msagent.py](tools/data_prepare/merge_msagent.py)

## 微调

```bash
mkdir -p /root/OculiChatDA/data
ln -s /share/model_repos/internlm2-chat-7b /root/OculiChatDA/data
xtuner list-cfg
mkdir config
xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 config/internlm2_chat_7b_qlora_oasst1_e3.py
cd config
mv internlm2_chat_7b_qlora_oasst1_e3_copy.py  internlm2_chat_7b_qlora_med_dialog_e3_copy.py
vim internlm2_chat_7b_qlora_med_dialog_e5.py # 修改配置文件
---> pretrained_model_name_or_path=data/interlm2-chat-7b
---> max_epochs = 5
---> data_path = "./data/qa_data.json"
---> batch_size = 4
---> lr = 1e-5
---> evaluation_inputs = evaluation_inputs = ['青光眼诊断的三要素是什么？', '糖尿病和糖尿病视网膜病变有什么关系呢', "医生你好，我的视野中心有黑色阴影，这是为什么呢?"]
---> dataset=dict(type=load_dataset, path="json", data_files=dict(train=data_path)),
---> dataset_map_fn = None

xtuner train config/internlm2_chat_7b_qlora_med_dialog_e5_copy.py --deepspeed deepspeed_zero2
# 实测batch为4耗显存26G，需要开一个2 * 1/4的机器
```

## 初步性能测试

```bash
xtuner convert pth_to_hf config/internlm2_chat_7b_qlora_med_dialog_e3_copy.py work_dirs/internlm2_chat_7b_qlora_med_dialog_e3_copy/epoch_1.pth  ./hf
xtuner convert merge ./data/internlm2-chat-7b ./hf ./merged --max-shard-size 2GB
xtuner chat ./merged --prompt-template internlm2_chat
输入:你是谁?
```

## Agent调用能力

输入:

````
#############################################
你是一个可以调用外部工具的助手，可以使用的工具包括：
{'FundusDiagnosis': '一个眼底图像诊断的工具，\n可以诊断眼底图像中的病变类型，如青光眼、是否为糖尿病视网膜病变。\n输入为眼底图的图像路径，可以为本地地址，也可以为网络地址(链接)\n'}
如果使用工具请遵循以下格式回复：
```
Thought:思考你当前步骤需要解决什么问题，是否需要使用工具
Action:工具名称，你的工具必须从 [['FundusDiagnosis']] 选择
Action Input:工具输入参数
```
工具返回按照以下格式回复：
```
Response:调用工具后的结果
```
如果你已经知道了答案，或者你不需要工具，请遵循以下格式回复
```
Thought:给出最终答案的思考过程
Final Answer:最终答案
```
开始
````

继续输入:

```
我上传了一张图片，图片路径为/a/b/c.jpg q请判断我是否患有青光眼
```

模型输出:

```
Thought: 你好！根据你的描述，你上传了一张眼底图像，需要判断是否患有青光眼。请问你的眼底图像是什么类型的？是彩色还是黑白？是数码的还是胶片的？另外，你的眼底图像是否有病变？如果有病变，是什么类型的？请提供更多的信息，以便我更好地帮助你。
Action: FundusDiagnosis
Action Input: {"image_path": "/a/b/c.jpg"}<|im_end|>
```

由此可确定微调后的模型没有遗忘Agent调用的能力。

## 模型能力测试:

## lmdeploy部署

```bash
lmdeploy convert internlm2-chat-7b  ./merged
lmdeploy serve api_server ./workspace --server-name 0.0.0.0 --server-port 23333 --tp 1
```

## 模型上传

```bash
python model_upload/convert.py
openxlab login
cd merged
openxlab model create --model-repo='flyer/OculiChatDA' -s ./metafile.yml

```
