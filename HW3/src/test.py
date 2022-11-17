import json
from transformers import pipeline
import argparse
from tqdm import tqdm


def main(args):
    progress = tqdm(total=5494)
    summarizer = pipeline(
        "summarization", model=args.model, batch_size=4, device=0, max_length=30, do_sample=True, top_p=0.7, top_k=20)
    list = []
    with open(args.target_file) as f:
        for line in f:
            obj = json.loads(line)
            list.append(json.dumps({
                "title": summarizer(obj['text'])[0]['summary_text'],
                "id": obj['id']
            }))
            progress.update(1)
        with open(args.dest_file, 'w+') as fp:
            fp.write('\n'.join(list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', help='the model directory')
    parser.add_argument(
        '-t', '--target_file', help='the test file path')
    parser.add_argument(
        '-d', '--dest_file', help='the final file path')
    args = parser.parse_args()
    main(args)
