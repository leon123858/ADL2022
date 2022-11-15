import json
import argparse


def main(args):
    with open(args.data) as file:
        list = []
        for line in file:
            line = json.loads(line)
            if args.no_ans == True:
                list.append(json.dumps({
                    'summary': line['id'],
                    'text': line['maintext'],
                }))
            else:
                list.append(json.dumps({
                    'summary': line['id'],
                    'text': line['maintext'],
                    'title': line['title']
                }))
        with open(args.target, 'w+') as fp:
            fp.write('\n'.join(list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', help='the data(.jsonl) should be process to target format')
    parser.add_argument(
        '-t', '--target', help='the preprocess result for data')
    parser.add_argument(
        '-n', '--no_ans', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args)
