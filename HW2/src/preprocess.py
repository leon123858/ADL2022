import json

with open("../data/train.json") as train_file:
    train_list = json.load(train_file)
    list = []
    for item in train_list:
        tmp = {k: v for k, v in item.items()}
        tmp['label'] = item['paragraphs'].index(item['relevant'])
        list.append(tmp)
    with open('../cache/train_select_data.json', 'w+', encoding='utf-8') as fp:
        json.dump(list, fp, ensure_ascii=False)

with open("../data/valid.json") as valid_file:
    valid_list = json.load(valid_file)
    list = []
    for item in train_list:
        tmp = {k: v for k, v in item.items()}
        tmp['label'] = item['paragraphs'].index(item['relevant'])
        list.append(tmp)
    with open('../cache/valid_select_data.json', 'w+', encoding='utf-8') as fp:
        json.dump(list, fp, ensure_ascii=False)
