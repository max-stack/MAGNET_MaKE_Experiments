# @Author Max Wilson-Hebben

# Used to generate a large scale dataset from the MaKE generated output file.
# This considers all of the output problems geneated rather than just the first.

import torch
import json
import jieba
import sys
sys.path.insert(0, '../')
from program_constants import MAKE_GENERATED_OUTPUT, MAKE_PROCESSED_TESTSET

def process_make_generated_dataset(input_path, reference_set_path):

    questions = torch.load(input_path)

    formatted_questions = []

    with open(reference_set_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    id = 0
    for i in range(len(questions)):
        formatted_questions.append({})
        formatted_questions[i]["id"] = id
        original_text = questions[i][0][1:-4]
        segmented_problem = ' '.join(jieba.lcut(original_text))
        formatted_questions[i]["original_text"] = original_text
        formatted_questions[i]["segmented_text"] = segmented_problem
        formatted_questions[i]["equation"] = data[i]["equation"]
        formatted_questions[i]["ans"] = data[i]["ans"]
        id += 1

    with open("final_testset.json", 'w', encoding='utf-8') as file:
        json.dump(formatted_questions, file, ensure_ascii=False, indent=4)

    # remove id=55 - question has no numbers


process_make_generated_dataset(MAKE_GENERATED_OUTPUT, MAKE_PROCESSED_TESTSET)
