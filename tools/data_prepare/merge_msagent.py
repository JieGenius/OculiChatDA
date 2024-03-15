import json
import random

from modelscope.msdatasets import MsDataset
from xtuner.dataset.map_fns import msagent_react_map_fn, template_map_fn_factory


def main():
    dataset = MsDataset.load('/root/MSAgent-Bench')
    res = []
    for i in range(len(dataset['train'])):
        if i > 5000:
            break
        item = dataset['train'][i]
        item['conversations'] = str(item['conversations'])
        res.append(msagent_react_map_fn(item))
    orig = 'data/processed_data/med_dialog_v2.json'
    with open(orig) as f:
        data = json.load(f)

    random.shuffle(res)
    data = data + res[:1000]

    with open('data/processed_data/med_dialog_screen_merge_msagent.json',
              'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
