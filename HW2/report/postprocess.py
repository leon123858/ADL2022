#!/usr/bin/python
import json
import sys

pred_path = sys.argv[1]

with open("./cache/answer_test_result.json") as answer_result:
    answer_list = json.load(answer_result)
    with open(pred_path, 'w+', encoding='utf-8') as pred_file:
        lines = ["id,answer\n"]
        for i in answer_list:
            lines.append(
                "{},{}\n".format(i['id'], i['prediction_text'].replace(',', '')))
        pred_file.writelines(lines)
