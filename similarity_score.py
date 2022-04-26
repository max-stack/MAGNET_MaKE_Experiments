# @Author: Max Wilson-Hebben

from nltk.tokenize import word_tokenize
from string import digits

class PreprocessText:
    def __init__(self, original_text):
        self.original_text = original_text

    def preprocess(self):
        self.remove_numbers()
        self.convert_lower_case()
        self.remove_punctuation()
        self.remove_apostrophe()

    def remove_numbers(self):
        remove_digits = str.maketrans('', '', digits)
        self.original_text = self.original_text.translate(remove_digits)

    def convert_lower_case(self):
        # lower case
        self.original_text = self.original_text.lower()

    def remove_punctuation(self):
        # remove punctuation
        symbols = "!\"#$%&()*+-.,/:;<=>?@[\]^_`{|}~\n"
        for symbol in symbols:
            self.original_text = self.original_text.replace(symbol, '')

    def remove_apostrophe(self):
        # remove apostrophe
        self.original_text = self.original_text.replace("'", "")


# Used to calculate the cosine similarity between 2 given problems
class Similarity_Ratio:
    def __init__(self, problem1, problem2):
        self.problem1 = problem1
        self.problem2 = problem2

    def process_problem(self, problem):
        problem_list = word_tokenize(problem)
        problem_set = {w for w in problem_list}
        return problem_set

    def cosine_similarity(self):
        preprocess_problem1 = PreprocessText(self.problem1)
        preprocess_problem1.preprocess()
        processed_problem1 = preprocess_problem1.original_text
        preprocess_problem2 = PreprocessText(self.problem2)
        preprocess_problem2.preprocess()
        processed_problem2 = preprocess_problem2.original_text
        X_set = self.process_problem(processed_problem1)
        Y_set = self.process_problem(processed_problem2)

        l1 = []
        l2 = []

        rvector = X_set.union(Y_set)

        for w in rvector:
            if w in X_set:
                l1.append(1)
            else:
                l1.append(0)
            
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        c = 0

        for i in range(len(rvector)):
            c += l1[i]*l2[i]

        cosine = c / float((sum(l1)*sum(l2))**0.5)

        return cosine

