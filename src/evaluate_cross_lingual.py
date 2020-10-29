import json
import pandas
import numpy as np
import random
from glob import glob
from src.data import * 
from src.bert import build_bert, get_bert
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

pretrained_model = 'bert-base-multilingual-cased'
# pretrained_model = 'xlm-roberta-base'

def prepare_set(_set):
    return zip(*list(map(lambda i: (i["text"], 1 if i["label"] == 1 else 0), _set)))

lang_sets = {
    "en": [ "india", "southafrica", "china" ],
    "es": [ "argentina" ],
    "pipeline": [ "india" ],
    "pt": [ "brazil" ],
    "tr": [ "turkey" ]
}

def read_lang(l, shuffle_it=True):
    files = glob("cross_lingual_data/" + l + "/*/*.json")
    sample_set = sum([ read_set(f) for f in files ], [])
    random.shuffle(sample_set)
    return sample_set


def get_no_training_predictions():

    print("-"*40)
    print("Random labeling evaluation")
    print("-"*40)

    model = get_bert(pretrained_model)

    for l in lang_sets:
        s = read_lang(l)
        x, y = prepare_set(s)
        preds = model.predict(s)
        print("Language:", l)
        print("-"*30)
        print(classification_report(y, preds))


def cross_evaluate(train_langs, test_lang):
    print("-"*40)
    print("Cross evaluation")
    print("Train language(s):", train_langs)
    print("Dev and Test language(s):", test_lang)
    print("-"*40)

    trn = sum([ read_lang(l) for l in train_langs ], [])
    random.shuffle(trn)

    s = read_lang(test_lang)
    random.shuffle(s)
    _, y = prepare_set(s)
    dev, test = train_test_split(s,
                            stratify=y, 
                            test_size=0.30)

    _, y = prepare_set(trn)
    trn, train_set_test = train_test_split(trn,
                            stratify=y, 
                            test_size=0.10)

    model = build_bert(trn, dev, pretrained_model, n_epochs=8)

    _, y = prepare_set(test)
    preds = model.predict(test)
    print("Language:", test_lang)
    print("-"*30)
    print(classification_report(y, preds, digits=4))


    _, y = prepare_set(train_set_test)
    preds = model.predict(train_set_test)
    print("Language:", "EN")
    print("-"*30)
    print(classification_report(y, preds, digits=4))


def multilingual_training():
    print("-"*40)
    print("Multilingual training")
    print("-"*40)

    langs = [ (l, read_lang(l)) for l in lang_sets ]
    trn, dev, test = [], [], []

    for k, l in langs:
        random.shuffle(l)
        _, y = prepare_set(l)
        ltrn, ltest = train_test_split(l,
                                    stratify=y, 
                                    test_size=0.25)
        test.append(ltest)
        trn += ltrn[len(ltrn) // 8:]
        dev += ltrn[:len(ltrn) // 8]

    random.shuffle(trn)
    random.shuffle(dev)

    model = build_bert(trn, dev, pretrained_model)

    for s, (k, l) in zip(test, langs):
        x, y = prepare_set(s)
        preds = model.predict(s)
        print("Language:", k)
        print("-"*30)
        print(classification_report(y, preds, digits=4))


def evaluate(trn, dev, test):
    model = build_bert(trn, dev, pretrained_model)
    x, y = prepare_set(test)
    preds = model.predict(test)
    print("-"*30)
    print(classification_report(y, preds, digits=4))


def evaluate_monolingual(lang, model):
    lang = read_lang(lang)
    random.shuffle(lang)
    _, y = prepare_set(lang)
    trn_dev, test = train_test_split(lang,
                            stratify=y, 
                            test_size=0.10)

    _, y = prepare_set(trn_dev)
    trn, dev = train_test_split(trn_dev,
                            stratify=y, 
                            test_size=0.10)

    model = build_bert(trn, dev, model, n_epochs=8)
    x, y = prepare_set(test)
    preds = model.predict(test)
    print("-"*30)
    print(classification_report(y, preds, digits=4))

    return model


if __name__ == '__main__':
    # print("Train and Dev on English, test on Spanish\n" + "-"*50)
    # model = evaluate_monolingual("en", pretrained_model)
    # test = read_lang("es")
    # x, y = prepare_set(test)
    # preds = model.predict(test)
    # print("-"*30)
    # print(classification_report(y, preds, digits=4))

    cross_evaluate(["en"], "es")

    # print("İngilizce + %40 Clarin training, Clarin %30 development, Clarin %30 test.")
    # es_lang = read_lang("es")

    # _, y = prepare_set(es_lang)
    # trn_dev, test = train_test_split(es_lang,
    #                         stratify=y, 
    #                         test_size=0.3)

    # _, y = prepare_set(trn_dev)
    # trn, dev = train_test_split(trn_dev,
    #                         stratify=y, 
    #                         test_size=0.4)

    # trn = trn + read_lang("en")
    # random.shuffle(trn)
    # evaluate(trn, dev, test)

    # print("Test monolingual")
    # evaluate_monolingual("es", pretrained_model)
    # evaluate_monolingual("es", "dccuchile/bert-base-spanish-wwm-cased")

    # print("Testing different english models")

    # print("roberta-base")
    # evaluate_monolingual("en", "roberta-base")
    # print(pretrained_model)
    # evaluate_monolingual("en", pretrained_model)
    # print("bert-base-cased")
    # evaluate_monolingual("en", "bert-base-cased")
    # print("albert-xlarge-v2")
    # evaluate_monolingual("en", "albert-large-v2")

