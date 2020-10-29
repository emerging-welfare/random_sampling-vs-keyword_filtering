import json
import pandas
import numpy as np
import random
from itertools import combinations
from data import * 
from bert import build_bert
from evaluate_sampling_methods import get_regex
from sklearn.metrics import classification_report
import random
random.seed(200)

filtering_list = False#"all_eventstatus_makarov_mmad_wang_keywords" # "makarov_keywords"
downsample_random = True

# filtering_list, downsample_random = False, False # use all random sample, without any filtering or downsampling
# filtering_list, downsample_random = 'makarov_keywords', False # filter random sample, without any downsampling
filtering_list, downsample_random = 'makarov_keywords', True # filter random sample, downsample to size of makarov.

def prepare_set(_set):
    return zip(*list(map(lambda i: (i["text"], 1 if i["label"] == 1 else 0), _set)))

country_sets = {
    # "india": [ "AL/IndianExpress.json", "AL/TheHindu.json", "AL/TimesOfIndia.json", "R/NewIndianExpress.json" ],
    "india": [ "R/NewIndianExpress.json" ],
    # "china": [ "AL/Guardian.json",  "AL/RCV1CodeChina.json", "R/PeoplesDaily.json", "R/SCMP.json", "R/SCMPRandom.json" ],
    "china": [ "R/PeoplesDaily.json", "R/SCMPRandom.json", "R/SCMP.json" ],
    # "southafrica": [ "AL/Sabinet_AL.json", "R/Sabinet_R.json"]
    "southafrica": [ "R/Sabinet_R.json"]
}

if __name__ == '__main__':
    trainingsets = list(combinations(country_sets.keys(), 2)) + list(combinations(country_sets.keys(), 1))
    countries = set(country_sets.keys())

    for trainingset in trainingsets:
        
        print("Training on:", trainingset)
        trainfiles = sum([country_sets[k] for k in trainingset],[])
        train_sets = sum([read_set("data/" + f) for f in trainfiles], [])
        train_sets = np.array(train_sets)
        dev_size = int(len(train_sets) * 0.15)
        train, dev = train_sets[dev_size:], train_sets[:dev_size]
        print('Size of train and dev are:', len(train), len(dev))
        
        if filtering_list and not downsample_random: # filter only training data. no downsample, it is not random anyway.
            print("Filtering data...")
            regex = get_regex(filtering_list)
            train = [ s for s in list(train) if regex.search(s["text"]) ]
            dev = [ s for s in list(dev) if regex.search(s["text"]) ]
            
        if filtering_list and downsample_random: # filter only training data. no downsample, it is not random anyway.
            print("downsamling to the size of:", filtering_list)
            print('type of train and dev are:', type(train), type(dev))
            regex = get_regex(filtering_list)
            
            train_len = len([ s for s in list(train) if regex.search(s["text"]) ])
            train = random.sample(list(train), train_len)
#             train = {k:v for k, v in train.items() if k in train_keys}

            dev_len = len([ s for s in list(dev) if regex.search(s["text"]) ])
            dev = random.sample(list(dev), dev_len)
#             dev = {k:v for k, v in dev.items() if k in dev_keys}  
            print('After downsampling, size of train and dev are:', len(train), len(dev))
            print('After downsampling, type of train and dev are:', type(train), type(dev))
#         break
#         import sys
#         sys.exit()

        model = build_bert(train, dev, "bert-base-uncased")
        testset = countries.difference(trainingset)

        for c in testset:
            print("Testing on:", c)   
            test = sum([read_set("data/" + f) for f in country_sets[c]], [])
            preds = list(model.predict(test))
            _, y_test = prepare_set(test)

            print("-" * 20)
            print("\n".join(classification_report(y_test, preds).split("\n")[-3:-1]))
            print("-" * 20)


#     print("-" * 20)
#     print("Total Evaluation, merge all datasets. without any cross-countryness.")
#     print("-" * 20)

#     all_files = sum([country_sets[k] for k in country_sets],[])
#     all_sets = sum([read_set("data/" + f) for f in all_files], [])
#     all_sets = np.array(all_sets)
#     test_size = int(len(all_sets) * 0.20)
#     dev_size = int(len(all_sets) * 0.10)
#     train, dev, test = all_sets[:- dev_size - test_size], all_sets[ - dev_size - test_size : - test_size], all_sets[-test_size:]
#     model = build_bert(train, dev, "bert-base-uncased")
#     preds = list(model.predict(test))
#     _, y_test = prepare_set(test)
#     print("-" * 20)
#     print("\n".join(classification_report(y_test, preds).split("\n")[-3:-1]))
