#!/usr/bin/python
import json

with open("../data/context.json") as context_file:
    context_list = json.load(context_file)
    with open("../data/test.json") as test_file:
        test_list = json.load(test_file)
        print(len(test_list), "test data count")
        with open("../cache/select_test_result.json") as mid_result:
            train_list = json.load(mid_result)
            print(len(train_list), "train select result")
            list = []
            for i, item in enumerate(test_list):
                tmp = {
                    'context': context_list[item['paragraphs'][train_list[i]]],
                    'id': item['id'],
                    'question': item['question'],
                    'title': item['id']}
                list.append(tmp)
            print(len(list), "next stage have example count")
            with open('../cache/middle_test_result.json', 'w+', encoding='utf-8') as fp:
                json.dump(list, fp, ensure_ascii=False)
