import json
import argparse


def main(args):
    with open(args.data) as file:
        list = []
        for line in file:
            line = json.loads(line)
            if args.ans == False:
                list.append(json.dumps({
                    'text': line['maintext'],
                    'id': line['id']
                }))
            else:
                list.append(json.dumps({
                    'summary': line['title'],
                    'text': line['maintext'],
                }))
        with open(args.target, 'w+', encoding='utf-8') as fp:
            fp.write('\n'.join(list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', help='the data(.jsonl) should be process to target format')
    parser.add_argument(
        '-t', '--target', help='the preprocess result for data')
    parser.add_argument('--ans', action='store_true')
    parser.add_argument('--no-ans', dest='ans', action='store_false')
    parser.set_defaults(ans=True)
    args = parser.parse_args()
    main(args)
