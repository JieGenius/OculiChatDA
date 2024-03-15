import json
from collections import Counter

SYSTEM = """你是一名眼科专家，你需要解答患者的疑问，提供准确的回答，必要时，提醒患者及时挂号就医。
当患者对症状描述不清时，你需要循序渐进的引导患者，询问患者的症状，以便给出准确的诊断。
当患者上传眼底图时，你需调用Agent判断眼底图中是否存在青光眼和糖尿病视网膜病变，若存在，需要提醒患者及时就医。
"""

SYSTEM_MED = """你是一名专业的医生，你需要解答患者的疑问，提供准确的回答，必要时，提醒患者及时挂号就医。
"""


def judge_is_relative_to_eye(conversation):
    keywords = [
        '眼', '视力', '视网膜', '青光眼', '白内障', '糖尿病视网膜病变', '视野', 'OCTA', 'fundus'
    ]
    for item in conversation:
        for keyword in keywords:
            if keyword in (item['input'] + item['output']):
                return True
        # if "眼" in (item["input"] + item["output"]):
        #     return True
    return False


def main():
    qa_data_path = [
        'data/processed_data/ophthalmology_9th Edition.json',
        'data/processed_data/ophthalmology_qa.json',
        'data/processed_data/中医眼科.json'
    ]
    med_dialog_path = 'data/raw_data/MedDialog/train_data.json'
    # universal_corpus_path = 'data/raw_data/part-006853-a894b46e.jsonl'

    final_res = []
    counter = Counter()
    for path in qa_data_path:
        counter_item = Counter()
        with open(path) as f:
            data = json.load(f)
        for item in data:
            for i in range(len(item['conversation'])):
                item['conversation'][i]['system'] = SYSTEM
            final_res.append(item)
            counter[len(item['conversation'])] += 1
            counter_item[len(item['conversation'])] += 1
        print(path, '的对话长度分布：', counter_item)
    print('自定义qa的对话长度分布：', counter)
    print('自定义qa的对话长度总数：', sum(counter.values()))
    counter_med_dialog = Counter()
    with open(med_dialog_path) as f:
        data = json.load(f)
    for item in data:
        tmp = {'conversation': []}
        assert len(item) % 2 == 0
        for i in range(0, len(item), 2):
            tmp['conversation'].append({
                'input': item[i].lstrip('病人：'),
                'output': item[i + 1].lstrip('医生：')
            })
        tmp['conversation'][0]['system'] = SYSTEM_MED
        counter_med_dialog[len(tmp['conversation'])] += 1
        # 判断是否和眼睛相关
        flag = judge_is_relative_to_eye(tmp['conversation'])
        if flag:
            final_res.append(tmp)
    print(tmp)
    print('med_dialog的对话长度分布：', counter_med_dialog)
    print('med_dialog的对话长度总数：', sum(counter_med_dialog.values()))
    #
    # med_conversation_num = sum(counter_med_dialog.values()) + sum(
    #     counter.values())
    # line_num = sum(1 for line in open(universal_corpus_path, "r"))
    # universal_corpus_num = min(9 * med_conversation_num,
    #                            500000, line_num)
    # with open(universal_corpus_path, "r") as f:
    #     for i in range(universal_corpus_num):
    #         tmp = json.loads(f.readline())
    #         final_res.append(
    #             {
    #                 "conversation": [
    #                     {
    #                         "output": tmp["content"]
    #                     }
    #                 ]
    #             }
    #         )
    #
    # print("universal_corpus的对话长度总数：", universal_corpus_num)
    print('总对话长度：', len(final_res))
    with open('data/processed_data/qa_data_eye.json', 'w') as f:
        json.dump(final_res, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
