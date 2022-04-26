# @Author Max Wilson-Hebben

# Program used to Chinese MaKE train, test and valid sets in a format ready for MWP solver models.

# Segmentation was required as well as solving the simultaneous equations since answers aren't - 
# given in the data.

import sympy as sym
import re
import ast
import operator as op
import json
import jieba
import sys
sys.path.insert(0, '../')
from program_constants import TRAIN_EQUATION1_PATH, TRAIN_EQUATION2_PATH, TRAIN_ORIGINAL_TEXT_PATH
from program_constants import TEST_EQUATION1_PATH, TEST_EQUATION2_PATH, TEST_ORIGINAL_TEXT_PATH
from program_constants import VALID_EQUATION1_PATH, VALID_EQUATION2_PATH, VALID_ORIGINAL_TEXT_PATH
from program_constants import MAKE_PROCESSED_TRAINSET, MAKE_PROCESSED_TESTSET, MAKE_PROCESSED_VALIDSET

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

class Create_Make_Datasets:
    def __init__(self, equation1_path, equation2_path, original_text_path, dataset_path, answers_path=None):
        self.equation1_path = equation1_path
        self.equation2_path = equation2_path
        self.original_text_path = original_text_path
        self.dataset_path = dataset_path
        self.answers_path = answers_path

    def create_equations(self):
        equations = []
        with open(self.equation1_path) as file:
            equation1_data = file.readlines()

        with open(self.equation2_path) as file:
            equation2_data = file.readlines()

        if(len(equation1_data) != len(equation2_data)):
            raise Exception("Number of equations don't match")

        for i in range(len(equation1_data)):
            equation1 = equation1_data[i][:-1].replace("x", "m")
            equation1 = equation1.replace("y", "n")
            equation2 = equation2_data[i][:-1].replace("x", "m")
            equation2 = equation2.replace("y", "n")
            final_equation1 = ""
            final_equation2 = ""
            for j in range(len(equation1)):
                if (equation1[j] == "m" or equation1[j] == "n") and j != 0 and equation1[j-1].isdigit():
                    final_equation1 += "*"
                final_equation1 += equation1[j]
            for j in range(len(equation2)):
                if (equation2[j] == "m" or equation2[j] == "n") and j != 0 and equation2[j-1].isdigit():
                    final_equation2 += "*"
                final_equation2 += equation2[j]
            equations.append([final_equation1, final_equation2])
        return equations

    def create_original_text(self):
        with open(self.original_text_path, encoding='utf-8') as file:
            original_text = file.readlines()

        for i in range(len(original_text)):
            original_text[i] = original_text[i][:-1]

        return original_text

    # Segmenting text puts spaces between words (jieba used for Chinese)
    def create_segmented_text(self, original_text):
        segmented_text = []
        for problem in original_text:
            segmented_problem = ' '.join(jieba.lcut(problem))
            segmented_text.append(segmented_problem)

        return segmented_text

    # Used to solve simultaneous equations and find an answer
    def generate_answers(self, equations):
        answers = []
        for equation_set in equations:
            try:
                prefix = equation_set[0].split("m")[0]
                if prefix[-1] == '-':
                    x1_coefficient = -1
                else:
                    x1_coefficient = float(prefix[:-1])
            except:         
                x1_coefficient = 1

            try:
                prefix = equation_set[1].split("m")[0]
                if prefix[-1] == '-':
                    x2_coefficient = -1
                else:
                    x2_coefficient = float(prefix[:-1])
            except:
                x2_coefficient = 1

            try:
                prefix = re.search('\+(.*)n', equation_set[0]).group(1)
                if prefix[-1] == '-':
                    y1_coefficient = -1
                else:
                    y1_coefficient = float(prefix[:-1])
            except:
                    y1_coefficient = 1

            try:
                prefix = re.search('\+(.*)n', equation_set[1]).group(1)
                if prefix[-1] == '-':
                    y2_coefficient = -1
                else:
                    y2_coefficient = float(prefix[:-1])
            except:
                y2_coefficient = 1

            answer1 = self.evaluate_answer(equation_set[0].split("=")[1])
            answer2 = self.evaluate_answer(equation_set[1].split("=")[1])

            x,y = sym.symbols('x,y')
            eq1 = sym.Eq(x1_coefficient*x + y1_coefficient*y, answer1)
            eq2 = sym.Eq(x2_coefficient * x + y2_coefficient * y, answer2)
            result = sym.solve([eq1, eq2], (x,y))
            answers.append(result)

        return answers

    def evaluate_answer(self, expression):
        return self.eval(ast.parse(expression, mode='eval').body)

    def eval(self, node):
        if isinstance(node, ast.Num): 
            return node.n
        elif isinstance(node, ast.BinOp): 
            return operators[type(node.op)](self.eval(node.left), self.eval(node.right))
        elif isinstance(node, ast.UnaryOp): 
            return operators[type(node.op)](self.eval(node.operand))
        else:
            raise TypeError(node)

    def create_dataset(self):
        dataset = []
        equations = self.create_equations()
        original_text = self.create_original_text()
        segmented_text = self.create_segmented_text(original_text)
        answers = self.generate_answers(equations)

        if len(equations) != len(original_text):
            raise Exception("Number of equations not equal to number of problem texts")

        if len(equations) != len(answers):
            raise Exception("Number of equations not equal to number of answers to problems")

        for i in range(len(equations)):
            dataset.append({})
            dataset[i]["id"] = i
            dataset[i]["original_text"] = original_text[i]
            dataset[i]["segmented_text"] = segmented_text[i]
            dataset[i]["equation"] = equations[i][0] + " ; " + equations[i][1]
            dataset[i]["ans"] = [float(answers[i][sym.Symbol('x')]), float(answers[i][sym.Symbol('y')])]

        with open(self.dataset_path, 'w', encoding='utf-8') as file:
            json.dump(dataset, file, ensure_ascii=False, indent=4)



train = Create_Make_Datasets(TRAIN_EQUATION1_PATH, TRAIN_EQUATION2_PATH, TRAIN_ORIGINAL_TEXT_PATH, MAKE_PROCESSED_TRAINSET)
test = Create_Make_Datasets(TEST_EQUATION1_PATH, TEST_EQUATION2_PATH, TEST_ORIGINAL_TEXT_PATH, MAKE_PROCESSED_TESTSET)
valid = Create_Make_Datasets(VALID_EQUATION1_PATH, VALID_EQUATION2_PATH, VALID_ORIGINAL_TEXT_PATH, MAKE_PROCESSED_VALIDSET)
# remove id=160 Valid datatset problem - has no numbers

train.create_dataset()
test.create_dataset()
valid.create_dataset()
