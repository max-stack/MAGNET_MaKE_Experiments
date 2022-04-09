import torch
import json
import jieba

questions = torch.load("MaKE_wo_plan_res.pt")

formatted_questions = []

with open("testset.json", 'r', encoding='utf-8') as file:
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

#remove id=55
