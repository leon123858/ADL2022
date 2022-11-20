import json
from transformers import pipeline
import argparse


def main(args):
    list = []
    with open(args.target_file) as tf:
        with open(args.origin_file) as of:
            for line in of:
                obj = json.loads(line)
                list.append(json.dumps({
                    "title": tf.readline(),
                    "id": obj['id']
                }))
            with open(args.dest_file, 'w+') as df:
                df.write('\n'.join(list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--origin_file', help='the origin test file path')
    parser.add_argument(
        '-t', '--target_file', help='the test result file path')
    parser.add_argument(
        '-d', '--dest_file', help='the final file path')
    args = parser.parse_args()
    main(args)
