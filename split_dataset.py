
'''
Dataset Splitter
'''

import json
import random

def split_dataset(input_file, train_file, val_file, train_ratio=0.8):
    '''Splits a JSONL file into training and validation sets.

    Args:
        input_file (str): Path to the input JSONL file.
        train_file (str): Path to the output training JSONL file.
        val_file (str): Path to the output validation JSONL file.
        train_ratio (float, optional): Ratio of training data. Defaults to 0.8.
    '''
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created {train_file} with {len(train_data)} lines.")
    print(f"Created {val_file} with {len(val_data)} lines.")

if __name__ == '__main__':
    split_dataset('resume_edge_cases_finetune.jsonl', 'train_edge_cases.jsonl', 'val_edge_cases.jsonl')
