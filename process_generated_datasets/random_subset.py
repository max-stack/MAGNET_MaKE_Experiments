# @Author: Max Wilson-Hebben

# Used to generate a random sample of problems from a larger dataset.

from random import seed, sample
import json
import sys
sys.path.insert(0, '../')
from program_constants import MAKE_GENERATED_TESTSET, RANDOM_SUBSET


class Random_Subset:
    def __init__(self, cur_seed, subset_size, dataset_path, subset_path):
        self.cur_seed = cur_seed
        self.subset_size = subset_size
        self.dataset_path = dataset_path
        self.subset_path = subset_path

    def generate_random_subset(self):
        seed(self.cur_seed)
        subset = []
        dataset_file = open(self.dataset_path, 'r', encoding="utf-8")
        dataset = json.load(dataset_file)
        indexes = sample(range(0,len(dataset)), self.subset_size)

        for i in range(self.subset_size):
            subset.append(dataset[indexes[i]])

        with open(self.subset_path, 'w', encoding="utf-8") as json_file:
            json.dump(subset, json_file, ensure_ascii=False, indent=4)


test = Random_Subset(93982136967, 20, MAKE_GENERATED_TESTSET, RANDOM_SUBSET)
test.generate_random_subset()
