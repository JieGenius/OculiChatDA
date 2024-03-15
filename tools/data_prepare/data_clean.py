import json

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from tqdm import tqdm

from tools.data_prepare.merge import judge_is_relative_to_eye

PROMPT = """你是一名数据筛选工程师，这些数据为医生和患者的对话数据，部分数据包含个人信息，还有一些数据为无用的数据，或者低质量的数据。
当数据符合要求时，你需要输出"合格"，否则输出"不合格"。用户输入时，D代表医生，P代表患者。
当输入的问诊数据和**眼科**不想关时，你也需要输出"不合格"。
下面是一些示例。
## 示例1:
Input:
   P: 糖尿病现眼底出血，我父亲61岁，患糖尿病已10来年了，这段时间老是觉得左眼角内侧磨，9月28日到河南省中医二附院眼科检查，发现眼底出血，大夫开了点吃的药。谢谢苏大夫百忙之中给我回复>，我会带着父亲在10月12号去门诊找您，到时再详谈，再次感谢！！
   D: 不客气！我会尽力！减轻患者痛苦是快乐的！
Output:
    不合格
Reason:
    缺乏上下文，属于低质量数据

## 示例2:
Input:
    P: 尿道下裂，小儿现19个月，自出生至现在，小便都不是射出来的，有时候急的话是流的，不急的话是滴下来的，所以这么大了，我们都没有真正把过尿；当地医院做过膀胱尿残留B超和尿常规，均无>发现问题，医生也叫我们继续观察。请问余大夫，要进行哪些检查才能确定为尿道下裂呢？并且能确定下裂的种类吗？
    D: 不一定是尿道下裂，要来检查。
    P: 具体要做什么检查呢？我们在宁波，去下上海有些不便，我们当地的宁大附属医院应该也可以检查。
    D: 要看过才能决定做什么检查，你可以去当地医院随访。
    P: 谢谢，知道了，如果是下裂，医生可以通过肉眼初步诊断吗？
    D: 是的，可以发现。
Output:
    不合格
Reason:
    与眼科无关

## 示例3:
Input:
    P: 远视+散光，2010.12.24学校体检时发现的。平时看电视看得很近。曾在武警浙江总队医院就诊过。麻请戴医生看下我家小孩的初查结果，请赐教！！谢谢您！！戴医生我是预约了你1月12号上午的门
诊，不知道还需不需要再散瞳，因为你比较忙可能还没空看到我上述的资料。如果还需要散瞳的话，麻请你看到资料告诉一下，因为我怕12号去门诊就诊来不及。。。谢谢。
    D: 抱歉，刚有空看到您的信息，您的孩子是远视散光，散光度数偏高，对视力会有影响，需要戴眼镜治疗，明天上午来我会给您的孩子快速散瞳验光，根据检查结果再给您解释。抱歉，刚有空看到您>的信息，您的孩子是远视散光，散光度数偏高，对视力会有影响，需要戴眼镜治疗，明天上午来我会给您的孩子快速散瞳验光，根据检查结果再给您解释。
Output:
    不合格
Reason:
    低质量数据，回复中有重复。并且包含医生个人信息/工作场所等。

## 示例4:
Input:
    P: 眼科，
    D: 请来医院面诊，我周四上午门诊.
Output:
    不合格
Reason:
    用户询问信息不完整，属于质量差的数据。医生直接要求去医院面诊，不满足互联网医疗的要求。

## 示例5:
Input:
    P: 眼病，2天。
    D: 可能是睑腺炎也叫麦粒肿，最好到眼科看下。
Output:
    不合格
Reason:
    用户询问信息不完整，根据医生的回复，用户可能上传了一张眼睛的图片，但是当前上下文未显示图片相关的描述，属于质量差的数据。

## 示例6:
Input:
    P:弱视，患者女,4岁半.弱视,一直在治疗,视力为何总是停留在0.6,就上不去了？我该怎么办？
    D:应该达到0.8，有些发育迟缓的也可以0.6，可以继续观察，或改为网络训练。
Output:
    合格

## 示例7:
Input:
    P: 医生你好，我眼睛有红血丝是什么原因？
    D: 眼睛出现红血丝可能有多种原因，其中一些常见的包括：
        眼睛疲劳：长时间使用电子设备，如电脑、手机或电视，或者长时间阅读，都可能导致眼睛疲劳和红血丝。
        干眼症：眼睛表面缺乏足够的泪液润滑可能导致眼睛干燥和红血丝。
        环境因素：暴露在干燥、尘土、烟雾或刺激性化学物质等环境中可能刺激眼睛，导致红血丝。
        过敏性结膜炎：对花粉、宠物毛发、灰尘或其他过敏原的过敏反应可能导致眼睛发红、瘙痒和红血丝。
        眼部感染：细菌或病毒感染可能引起眼睛发红、疼痛和红血丝。
        眼部损伤：眼睛受到外伤或划伤可能导致出血和红血丝。
        如果您的眼睛经常出现红血丝，建议您尽快就医，以便医生进行全面的眼部检查，并根据具体情况提供治疗建议。
    P：如果是眼睛疲劳导致的，多长时间可恢复？
    D：眼睛疲劳导致的红血丝通常在适当的休息和护理下会逐渐缓解。恢复时间取决于疲劳的程度以及您采取的措施。一般来说，如果您采取以下措施，通常可以在几个小时到一天内看到改善：
        休息眼睛：停止使用电子设备或进行长时间的阅读，让眼睛得到充分休息。
        使用湿润眼液：使用适当的人工泪液来帮助缓解眼睛干燥和不适感。
        调整环境：确保工作或学习环境充足的光线，避免过度暴露在干燥、尘土或刺激性化学物质的环境中。
        进行眼部按摩：轻轻按摩眼部周围的肌肉有助于缓解眼睛疲劳。
        使用冷敷：将湿润的冷毛巾或冷水瓶轻轻敷在闭着的眼睛上有助于减轻眼睛的疲劳感。
        如果您的眼睛持续感到疲劳和不适，或者红血丝没有得到改善，建议您寻求眼科医生的进一步评估和治疗建议。
Output:
    合格

开始！！！
"""

SYSTEM_MED = """你是一名眼科专家，你需要解答患者的疑问，并且给出诊断和治疗建议。
"""


def main():
    backend_config = TurbomindEngineConfig(tp=2)
    gen_config = GenerationConfig(
        top_p=0.8, top_k=40, temperature=0.8, max_new_tokens=1024)
    # lmdeploy convert internlm2-chat-20b ~/share/model_repos/internlm2-chat-20b/ --dst-path workspace_20b
    # pipe = pipeline('/root/OculiChatDA/workspace_20b',model_name='internlm2-chat-20b',
    pipe = pipeline('InternLM/internlm2-chat-20b-4bits', model_format='awq')
    # ,model_name='internlm2-chat-20b',
    #                     backend_config=backend_config
    prompts = [{
        'role': 'system',
        'content': PROMPT
    }, {
        'role': 'user',
        'content': 'Shanghai is'
    }]
    med_dialog_path = 'data/raw_data/MedDialog/train_data.json'

    with open(med_dialog_path) as f:
        data = json.load(f)
    final_res = []
    valid_cnt = 0
    invalid_cnt = 0
    processed = 0
    total = len(data)
    for item in tqdm(data):
        processed += 1
        tmp = {'conversation': []}
        assert len(item) % 2 == 0
        for i in range(0, len(item), 2):
            tmp['conversation'].append({
                'input': item[i].lstrip('病人：'),
                'output': item[i + 1].lstrip('医生：')
            })
        flag = judge_is_relative_to_eye(tmp['conversation'])
        if flag:
            inp = ''
            for i in range(len(tmp['conversation'])):
                inp += 'P: ' + tmp['conversation'][i]['input'] + '\n'
                inp += 'D: ' + tmp['conversation'][i]['output'] + '\n'
            prompts[1]['content'] = inp
            response = pipe(prompts, gen_config=gen_config)
            response = response.text
            if response[:2] == '合格':
                valid_cnt += 1
                tmp['conversation'][0]['system'] = SYSTEM_MED
                final_res.append(tmp)
            else:
                invalid_cnt += 1
        print(
            'total: %d, processed: %d, remained: %d, valid: %d, invalid: %d' %
            (total, processed, total - processed, valid_cnt, invalid_cnt))
        if (valid_cnt % 1000 == 0) and (valid_cnt != 0):
            with open('data/processed_data/med_dialog_v2.json', 'w') as f:
                json.dump(final_res, f, indent=2, ensure_ascii=False)
    with open('data/processed_data/med_dialog_v2.json', 'w') as f:
        json.dump(final_res, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
