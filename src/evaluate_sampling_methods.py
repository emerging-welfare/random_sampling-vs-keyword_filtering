from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.svm import SVC
from recall_biased_estimator import RecallBiasedEstimator
from bert import build_bert
from data import fold_iterator_sklearn, read_set
from glob import glob
import data
import re
import random
random.seed(100)

classifiers = {}

def prepare_set(_set):
    return zip(*list(map(lambda i: (i["text"], 1 if i["label"] == 1 else 0), _set)))

def get_regex(list_file):
    with open("keyterm_lists/" + list_file, "r") as f:
        regex = f.read().splitlines()

    for i,key in enumerate(regex):
        if i == 0:
            reg = key
        else:
            reg = reg + "|" + key

    return re.compile(reg, re.I)

def build_pipeline(train, dev, clf):
    x_train, y_train = prepare_set(train)
    x_dev, y_dev = prepare_set(dev)

    bow = CountVectorizer(max_features=2000)
    tfidf = TfidfTransformer()
    
    if clf == "SVM":
        clf = SVC(C=10, gamma=1, kernel="rbf")
    elif clf == "RF":
        clf = RandomForestClassifier(n_estimators=25, max_depth=35)
    elif clf == "NB":
        clf = ComplementNB(norm=False)
    elif clf == "R":
        clf = RecallBiasedEstimator([SVC(C=10, gamma=1, kernel="rbf"), RandomForestClassifier(n_estimators=10, max_depth=35), ComplementNB(norm=False)])

    pipeline = Pipeline([('bow', bow),
                        ('tfidf', tfidf),
                        ('clf', clf)])

    pipeline.fit(x_train, y_train)

    return pipeline

def evaluate_sampling_method(path, filter=None, filtering_type="pre", size=-1):
    global classifiers
    files = glob(path + "/*.json")
    all_samples_orig = sum([read_set(f) for f in files], [])
    all_samples = []
    for s_dict in all_samples_orig:
        tmp_dict = {}
        for k, v in s_dict.items():
            if k == 'label' and v == 2:
                tmp_dict[k] = 0
            else:
                tmp_dict[k] = v
                
        all_samples.append(tmp_dict)
                
#     print('type of first item of all_samples:', type(all_samples[0]))
#     print('first item of all_samples:', all_samples[0])
#     print("Total Number of", len(all_samples), "Samples")
#     print("Type of all_samples:", type(all_samples))
    
    if filter and filtering_type == "pre" and size == -1: # filter both test and training data
        print("Filtering data... pre")
        regex = get_regex(filter)
        all_samples = [ s for s in all_samples if regex.search(s["text"]) ]
        print("After filtering we get", len(all_samples), "Samples")
        
    # No filter. But downsample random dataset accirding to size of a filter.
    if filter and filtering_type == "downsample_random" and size == -1: 
        print('filter is:', filter, '-- filter_type is downsample_random.')
        regex = get_regex(filter)
        all_samples = random.sample(all_samples, len([ s for s in all_samples if regex.search(s["text"])]))
        print("After downsampling we get", len(all_samples), "Samples")
        
    if size > 0 and filter is None: # No filter. But downsample random dataset.
        print('-- downsample by size. The size is:', str(size))
        print('The filter is None. The random sample will be downsized.')
        all_samples = random.sample(all_samples, size)
        print("After downsampling by size we get", len(all_samples), "Samples")

    all_true, all_probs = [], { c : [] for c in classifiers.keys() }
    for i, (train, dev, test) in enumerate(fold_iterator_sklearn(all_samples, K=5)):

        # x, y = prepare_set(train)
        # train, _, _, _ = train_test_split(train, y, test_size=0.75)
        # x, y = prepare_set(dev)
        # dev, _, _, _ = train_test_split(dev, y, test_size=0.75)

        if filter and filtering_type == "post" and size == -1: # filter only training data
            print("Filtering data... post")
            regex = get_regex(filter)
            train = [ s for s in train if regex.search(s["text"]) ]
            dev = [ s for s in dev if regex.search(s["text"]) ]
            print("After filtering we get: train, dev, test, total:", len(train), len(dev), len(test), len(train)+ len(dev)+len(test)) # output for makarov: 1400, 149, 1,192, 2,741
            
        if filter and filtering_type == "post" and size > 0: # filter only training data
            print("Filtering data... post and downsample, size will be", str(size))
            
            regex = get_regex(filter)
            
            print("train and dev lengths:", len(train), len(dev))
            print("train and dev lengths should be after resampling:", int(size*0.9), int(size*0.1))
            print("train actual length after filtering:", len([ s for s in train if regex.search(s["text"]) ]))
            print("dev actual length after filtering:", len([ s for s in dev if regex.search(s["text"]) ]))
            
            if int(size*0.9) < len([ s for s in train if regex.search(s["text"]) ]):
                print('train becomes size*0.9')
                train = random.sample([ s for s in train if regex.search(s["text"]) ], int(size*0.9))
            else:
                print('train is not downsampled.')
                train = [ s for s in train if regex.search(s["text"]) ]
            
            if int(size*0.1) < len([ s for s in dev if regex.search(s["text"]) ]):
                print('dev becomes size*0.1')
                dev = random.sample([ s for s in dev if regex.search(s["text"]) ], int(size*0.1))
            else:
                print('dev is not downsampled.')
                dev = [ s for s in dev if regex.search(s["text"]) ]
                
            print("After filtering and downsampling we get: train, dev, test, total:", len(train), len(dev), len(test), len(train)+ len(dev)+len(test)) # output for makarov: 1400, 149, 1,192, 2,741

        if filter and filtering_type == "test": # ????
            print("Filtering data... test")
            regex = get_regex(filter)
            test = [ s for s in test if regex.search(s["text"]) ]
            x, y = prepare_set(train)
            t_len = len([s for s in train if regex.search(s["text"]) ]) 
            train, _ = train_test_split(train, stratify=y, train_size=t_len)
            x, y = prepare_set(dev)
            d_len = len([s for s in dev if regex.search(s["text"]) ])
            dev, _ = train_test_split(dev, stratify=y, train_size=d_len)

        x_test, y_test = prepare_set(test)
        all_true += y_test
        
#         print('The fold is:', i)
#         print('The type of test is:', type(test))
#         print('The test is:', test)
        
#         print('The type of x_test is:', type(x_test))
#         print('The x_test is:', x_test)
#         continue

        print("Before starting training, we have: train, dev, test, total:", len(train), len(dev), len(test), len(train)+ len(dev)+len(test))
        for c, build_classifier in classifiers.items():
            classifier = build_classifier(train, dev, c)
            all_probs[c] += list(classifier.predict(test if c == "bert-base-uncased" else x_test))
            del classifier
            
    print("Final Results of Cross Validation:", filter, str(filtering_type))
    print("-" * 40)
    print()
    
    for c, preds in all_probs.items():
        print("Method:", c)
        print("-" * 20)
        print("\n".join(classification_report(all_true, preds, digits=4).split("\n")[-3:-1]))

def main():
    global classifiers
    classifiers = {"SVM": build_pipeline, "NB": build_pipeline, "RF": build_pipeline, "R": build_pipeline, "bert-base-uncased": build_bert}
    # classifiers = {"bert-base-uncased": build_bert}

    # print("-" * 40, "\nInternal Evaluation of Bert\n" + ("-" * 40))
    # evaluate_sampling_method("data/R", filter="wang_keywords", filtering_type="pre")
    # evaluate_sampling_method("data/R", filter="makarov_keywords", filtering_type="pre")
    # evaluate_sampling_method("data/R", filter="mmad_keywords", filtering_type="pre")
    # evaluate_sampling_method("data/R", filter="eventstatus_keywords", filtering_type="pre")
    # print("-" * 40, "\nUniversal Evaluation of Bert\n" + ("-" * 40))
    # evaluate_sampling_method("data/R", filter="wang_keywords", filtering_type="post")
    # evaluate_sampling_method("data/R", filter="makarov_keywords", filtering_type="post")
    # evaluate_sampling_method("data/R", filter="mmad_keywords", filtering_type="post")
    # evaluate_sampling_method("data/R", filter="eventstatus_keywords", filtering_type="post")
    
    # evaluate_sampling_method("data/R")

#     print("-" * 40, "\nEvaluation of resized random lists on restricted test sets\n" + ("-" * 40))
#     print("Evaluate, wang:evaluate_sampling_method(data/R, filter=wang_keywords, filtering_type=test)")
#     evaluate_sampling_method("data/R", filter="wang_keywords", filtering_type="test")
#     print("Evaluate, makarov:evaluate_sampling_method(data/R, filter=makarov_keywords, filtering_type=test)")
#     evaluate_sampling_method("data/R", filter="makarov_keywords", filtering_type="test")
#     print("Evaluate, mmad:evaluate_sampling_method(data/R, filter=mmad_keywords, filtering_type=test)")
#     evaluate_sampling_method("data/R", filter="mmad_keywords", filtering_type="test")
#     print("Evaluate, eventstatus:evaluate_sampling_method(data/R, filter=eventstatus_keywords, filtering_type=test)")
#     evaluate_sampling_method("data/R", filter="eventstatus_keywords", filtering_type="test")
#     print("Evaluate, all_eventstatus_makarov_mmad_wang:evaluate_sampling_method(data/R, filter=all_eventstatus_makarov_mmad_wang_keywords, filtering_type=test)")
#     evaluate_sampling_method("data/R", filter="all_eventstatus_makarov_mmad_wang_keywords", filtering_type="test")

#     print("-" * 40, "\nEvaluation with pre filtering\n" + ("-" * 40))
#     print("Evaluate, wang:evaluate_sampling_method(data/R, filter=wang_keywords, filtering_type=pre)")
#     evaluate_sampling_method("data/R", filter="wang_keywords", filtering_type="pre")
#     print("Evaluate, makarov:evaluate_sampling_method(data/R, filter=makarov_keywords, filtering_type=pre)")
#     evaluate_sampling_method("data/R", filter="makarov_keywords", filtering_type="pre")
#     print("Evaluate, mmad:evaluate_sampling_method(data/R, filter=mmad_keywords, filtering_type=pre)")
#     evaluate_sampling_method("data/R", filter="mmad_keywords", filtering_type="pre")
#     print("Evaluate, eventstatus:evaluate_sampling_method(data/R, filter=eventstatus_keywords, filtering_type=pre)")
#     evaluate_sampling_method("data/R", filter="eventstatus_keywords", filtering_type="pre")
#     print("Evaluate, all_eventstatus_makarov_mmad_wang:evaluate_sampling_method(data/R, filter=all_eventstatus_makarov_mmad_wang_keywords, filtering_type=pre)")
#     evaluate_sampling_method("data/R", filter="all_eventstatus_makarov_mmad_wang_keywords", filtering_type="pre")

#     print("-" * 40, "\nEvaluation with post filtering\n" + ("-" * 40))
#     print("Evaluate, wang:evaluate_sampling_method(data/R, filter=wang_keywords, filtering_type=post)")
#     evaluate_sampling_method("data/R", filter="wang_keywords", filtering_type="post")
#     print("Evaluate, makarov:evaluate_sampling_method(data/R, filter=makarov_keywords, filtering_type=post)")
#     evaluate_sampling_method("data/R", filter="makarov_keywords", filtering_type="post")
#     print("Evaluate, mmad:evaluate_sampling_method(data/R, filter=mmad_keywords, filtering_type=post)")
#     evaluate_sampling_method("data/R", filter="mmad_keywords", filtering_type="post")
#     print("Evaluate, eventstatus:evaluate_sampling_method(data/R, filter=eventstatus_keywords, filtering_type=post)")
#     evaluate_sampling_method("data/R", filter="eventstatus_keywords", filtering_type="post")
#     print("Evaluate, all_eventstatus_makarov_mmad_wang:evaluate_sampling_method(data/R, filter=all_eventstatus_makarov_mmad_wang_keywords, filtering_type=post)")
#     evaluate_sampling_method("data/R", filter="all_eventstatus_makarov_mmad_wang_keywords", filtering_type="post")
                             
#     print("-" * 40, "\nEvaluation without any filtering. but downsampling\n" + ("-" * 40))
#     print("Evaluate, wang size downsampling Random:evaluate_sampling_method(data/R)")
#     evaluate_sampling_method("data/R", filter="wang_keywords", filtering_type='downsample_random')  
#     print("Evaluate, makarov size downsampling Random:evaluate_sampling_method(data/R)")
#     evaluate_sampling_method("data/R", filter="makarov_keywords", filtering_type='downsample_random')  
#     print("Evaluate, mmad size downsampling Random:evaluate_sampling_method(data/R)")
#     evaluate_sampling_method("data/R", filter="mmad_keywords", filtering_type='downsample_random')  
#     print("Evaluate, eventstatus size downsampling Random:evaluate_sampling_method(data/R)")
#     evaluate_sampling_method("data/R", filter="eventstatus_keywords", filtering_type='downsample_random')
#     print("Evaluate, all_eventstatus_makarov_mmad_wang, size downsampling Random:evaluate_sampling_method(data/R)")
#     evaluate_sampling_method("data/R", filter="all_eventstatus_makarov_mmad_wang_keywords", filtering_type='downsample_random')  

#     print("-" * 40, "\nEvaluation without any filtering.\n" + ("-" * 40))
#     print("Evaluate, Random:evaluate_sampling_method(data/R)")
#     evaluate_sampling_method("data/R")
    
    for sz in range(625, 6000, 250):
        print("-" * 40, "\nEvaluation without any filtering. Size:" + str(sz) + '\n' + ("-" * 40))
        print("Evaluate, Random:evaluate_sampling_method(data/R)")
        evaluate_sampling_method("data/R", size=sz)
        
#     for sz in range(1500, 1625, 125):
#         print("-" * 40, "\nEvaluation with Makarov filtering. Size:" + str(sz) + '\n' + ("-" * 40))
#         print("Evaluate, Makarov:evaluate_sampling_method(data/R, filter=makarov_keywords size=sz)")
#         evaluate_sampling_method("data/R", filter="makarov_keywords", filtering_type='post', size=sz)
        
    



if __name__ == "__main__":
    main()