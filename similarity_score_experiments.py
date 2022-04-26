# Author: Max Wilson-Hebben

# A group of experiments used to generate data for the paper, using metrics and similarity score.

from similarity_score import Similarity_Ratio
from program_constants import MAKE_PROCESSED_TRAINSET, MAKE_PROCESSED_TESTSET, MAKE_PROCESSED_VALIDSET
from program_constants import MAKE_GENERATED_TESTSET
from rouge_score import rouge_scorer
from nltk.translate import bleu_score, meteor_score
import json
import numpy as np
import matplotlib.pyplot as plt

file = open(MAKE_GENERATED_TESTSET, encoding="utf-8")
data = json.load(file)

file = open(MAKE_PROCESSED_TRAINSET, encoding="utf-8")
train_data = json.load(file)

file = open(MAKE_PROCESSED_TESTSET, encoding="utf-8")
train_data.extend(json.load(file))

file = open(MAKE_PROCESSED_TESTSET, encoding="utf-8")
test_data = json.load(file)

file = open(MAKE_PROCESSED_VALIDSET, encoding="utf-8")
train_data.extend(json.load(file))


# Generates the highest similarity score of a problem over the entire train set.
def similarity_score(problem1, train_data):
    highest_similarity = 0
    for j in range(len(train_data)):
        test = Similarity_Ratio(problem1, train_data[j]["segmented_text"])
        similarity = test.cosine_similarity()
        if similarity > highest_similarity:
            highest_similarity = similarity
    
    return highest_similarity


def calculate_average_metric_scores():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    total_bleu = 0
    total_meteor = 0

    for i in range(len(data)):
        out_problem = data[i]["segmented_text"]
        dev_problem = test_data[i]["segmented_text"]
        cur_rouge1_score = scorer.score(dev_problem, out_problem)["rouge1"].fmeasure
        cur_rouge2_score = scorer.score(dev_problem, out_problem)["rouge2"].fmeasure
        cur_rougeL_score = scorer.score(dev_problem, out_problem)["rougeL"].fmeasure
        cur_bleu_score = bleu_score.sentence_bleu([dev_problem.split()], out_problem.split())
        cur_meteor_score = meteor_score.meteor_score([dev_problem.split()], out_problem.split())
        total_rouge1 += cur_rouge1_score
        total_rouge2 += cur_rouge2_score
        total_rougeL += cur_rougeL_score
        total_bleu += cur_bleu_score
        total_meteor += cur_meteor_score

    print("Rouge1: " + str(total_rouge1/len(data)))
    print("Rouge2: " + str(total_rouge2/len(data)))
    print("RougeL: " + str(total_rougeL/len(data)))
    print("BLEU: " + str(total_bleu/len(data)))
    print("METEOR: " + str(total_meteor/len(data)))


# Removes problems in the test set that have a similarity ratio > max_ratio
def remove_similar_problems_testset(max_ratio=0.9):
    test_file = open(MAKE_PROCESSED_TESTSET, encoding="utf-8")
    output_file = open(MAKE_GENERATED_TESTSET, "w", encoding="utf-8")

    new_problem_set = []
    new_dev_set = []
    for i in range(len(data)):
        max_ratio = similarity_score(data[i]["segmented_text"], train_data)
        print(max_ratio)
        if(max_ratio < max_ratio):
            new_problem_set.append(data[i])
            new_dev_set.append(test_data[i])

    json.dump(new_problem_set, output_file, ensure_ascii=False, indent=4)
    json.dump(new_dev_set, test_file, ensure_ascii=False, indent=4)


# Calculates individual metrics for each problem given
def calculate_individual_metric_scores():
    rouge_scores = []
    bleu_scores = []
    meteor_scores = []
    similarity = []

    i = 0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for i in range(len(data)):
        if test_data[i] is not None and data[i] is not None:
            tgt_problem = test_data[i]["segmented_text"]
            out_problem = data[i]["segmented_text"]
            cur_rouge_score = scorer.score(tgt_problem, out_problem)["rougeL"].fmeasure
            cur_bleu_score = bleu_score.sentence_bleu([tgt_problem.split()], out_problem.split())
            cur_meteor_score = meteor_score.meteor_score([tgt_problem.split()], out_problem.split())

            rouge_scores.append(cur_rouge_score)
            bleu_scores.append(cur_bleu_score)
            meteor_scores.append(cur_meteor_score)
            max_ratio = similarity_score(out_problem, train_data)
            similarity.append(max_ratio)
            print(max_ratio)
        i += 1

    sorted_rouge = np.array([x for _,x in sorted(zip(similarity,rouge_scores))])
    sorted_meteor = np.array([x for _,x in sorted(zip(similarity,meteor_scores))])
    sorted_bleu = np.array([x for _,x in sorted(zip(similarity,bleu_scores))])

    similarity.sort()
    similarity = np.array(similarity)

    return [similarity, sorted_rouge, sorted_bleu, sorted_meteor]


def generate_metric_graph():
    metric_values = calculate_individual_metric_scores()
    x = np.array(list(range(len(metric_values[1]))))

    plt.plot(x, metric_values[1], label="ROUGE Score")
    plt.plot(x, metric_values[2], label="BLEU Score")
    plt.plot(x, metric_values[3], label="METEOR Score")
    plt.plot(x, metric_values[0], label="Similarity Score")
    plt.xlabel("Generated Problem Index")
    plt.ylabel("Ratio Score")
    plt.legend()

    plt.show()


def generate_metric_scatter():
    metric_values = calculate_individual_metric_scores()
    
    x = np.array(list(range(len(metric_values[1]))))

    a_sim, b_sim = np.polyfit(x, metric_values[0], 1)
    a_rouge, b_rouge = np.polyfit(x, metric_values[1], 1)
    a_bleu, b_bleu = np.polyfit(x, metric_values[2], 1)
    a_met, b_met = np.polyfit(x, metric_values[3], 1)

    plt.scatter(x, metric_values[1], s=1)
    plt.scatter(x, metric_values[2], s=1)
    plt.scatter(x, metric_values[3], s=1)
    plt.scatter(x, metric_values[0], s=1)
    plt.plot(x, a_rouge*x+b_rouge, label="ROUGE Score")
    plt.plot(x, a_bleu*x+b_bleu, label="BLEU Score")
    plt.plot(x, a_met*x+b_met, label="METEOR Score")
    plt.plot(x, a_sim*x+b_sim, label="Similarity Score")
    plt.xlabel("Generated Problem Index")
    plt.ylabel("Ratio Score")
    plt.legend()

    plt.show()


def generate_similarity_histogram():
    y = [-1 for x in range(1000)]
    for i in range(1000):
        cur_problem = train_data[i]["segmented_text"]
        print(i)
        for j in range(len(train_data)):
            test = Similarity_Ratio(cur_problem, train_data[j]["segmented_text"])
            similarity = test.cosine_similarity()
            if similarity > 0.9:
                y[i] += 1

    bins = 9
    arr = plt.hist(y, bins=bins)
    plt.xticks([x for x in range(bins)])
    for i in range(bins):
        plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
    plt.xlabel("Number of Train Problems with Similarity > 0.9")
    plt.ylabel("Number of Math Word Problems")
    plt.title("A Graph to Show the Number of Training Problems with Similarity > 0.9 to a Subset of 100 Training MWPs")
    plt.show()


# Used to view a problem's most similar MWP in the trainset
def view_similar_questions():
    for i in range(len(data)):
        highest_similarity = 0
        similar_problem = ""
        cur_problem = data[i]["segmented_text"]
        for j in range(len(train_data)):
            test = Similarity_Ratio(cur_problem, train_data[j]["segmented_text"])
            similarity = test.cosine_similarity()
            if similarity > highest_similarity:
                highest_similarity = similarity
                similar_problem = train_data[j]["segmented_text"]
        input(highest_similarity)
        input(cur_problem)
        input(similar_problem)
