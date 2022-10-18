#!/usr/bin/python
import json
# select
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
# answer
with open("../data/context.json") as context_file:
    context_list = json.load(context_file)
    with open("../data/train.json") as train_file:
        train_list = json.load(train_file)
        list = []
        for item in train_list:
            tmp = {'answers': {'answer_start': [item['answer']['start']], 'text': [item['answer']['text']]},
                   'context': context_list[item['relevant']],
                   'id': item['id'],
                   'question': item['question'],
                   'title': item['id']}
            list.append(tmp)
        with open('../cache/train_answer_data.json', 'w+', encoding='utf-8') as fp:
            json.dump(list, fp, ensure_ascii=False)

    with open("../data/valid.json") as valid_file:
        valid_list = json.load(valid_file)
        list = []
        for item in train_list:
            tmp = {'answers': {'answer_start': [item['answer']['start']], 'text': [item['answer']['text']]},
                   'context': context_list[item['relevant']],
                   'id': item['id'],
                   'question': item['question'],
                   'title': item['id']}
            list.append(tmp)
        with open('../cache/valid_answer_data.json', 'w+', encoding='utf-8') as fp:
            json.dump(list, fp, ensure_ascii=False)
