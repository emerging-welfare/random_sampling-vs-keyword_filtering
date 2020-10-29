import pandas as pd
import re
from sklearn.metrics import classification_report
import argparse

global reg

def regsearch(row):
    global reg
    m = reg.search(row.text)

    if m:
        row.keyword_pred = 1 

    return row

def print_stuff(doc):
    doc.loc[doc.label == 2, 'label'] = 0
    print("Positive labels : " + str(len(doc[doc.label == 1])))
    print("Positive preds : " + str(len(doc[doc.keyword_pred == 1])))
    print("Full length : " + str(len(doc)) + "\n")

    print("Prediction Report\n")
    print(classification_report(doc.label, doc.keyword_pred))

def get_false_negatives(doc): 
    doc = pd.read_json(doc, orient="records", lines=True) 
    doc['keyword_pred'] = 0 
    doc = doc[doc['label'].isin([0.0,1.0,1,0,0,2,2.0])] 
    doc = doc.apply(regsearch, axis=1) 
    return doc[(doc["keyword_pred"] == 0) & (doc["label"] == 1)]

def get_false_positives(doc): 
    doc = pd.read_json(doc, orient="records", lines=True) 
    doc['keyword_pred'] = 0 
    doc = doc[doc['label'].isin([0.0,1.0,1,0,0,2,2.0])] 
    doc = doc.apply(regsearch, axis=1) 
    return doc[(doc["keyword_pred"] == 1) & (doc["label"] == 0)] 

def main():
    parser = argparse.ArgumentParser(description="This script takes a regex list and checks the match percent given input docs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r','--regex_list', type=str,help="Regex list, every line has a regex", action='store',default="",required=True)
    parser.add_argument('documents', nargs='+', help='Json lines docs with text and label')
    args = parser.parse_args()

    with open(args.regex_list, "r") as f:
        regex = f.read().splitlines()

    global reg
    for i,key in enumerate(regex):
        print(i, key)
        if i == 0:
            reg = key
        else:
            reg = reg + "|" + key


    reg = re.compile(reg, re.I)
    print('The file is:', args.regex_list)
    print('The regular expression is:', reg)
    print("---------------------------------------------------------------------------------------------------------")


    all_df = pd.DataFrame()

    for doc in args.documents:

        print(doc)
        
        doc_df = pd.read_json(doc, orient="records", lines=True)
        # remove lines that contain the urls
        doc_df['text'] = doc_df.text.apply(lambda d: '\n'.join([l for l in d.split('\n') if not (len(l)<200 and 'url' in l and 'http' in l)])) 
        # remove lines that contain the publication date.
        doc_df['text'] = doc_df.text.apply(lambda d: '\n'.join([l for l in d.split('\n') if not (len(l)<100 and re.search(r'\bJanuary\b|\bFebruary\b|\bMarch\b|\bApril\b|\bMay\b|\bJune\b|\bJuly\b|\bAugust\b|\bSeptember\b|\bOctober\b|\bNovember\b|\bDecember\b', l) and re.search(r'\d{4}',l))]))

 


        doc_df['keyword_pred'] = 0
        doc_df = doc_df[doc_df['label'].isin([0.0,1.0,1,0,0,2,2.0])]
        doc_df = doc_df.apply(regsearch, axis=1)
        all_df = all_df.append(doc_df, ignore_index=True)

        print_stuff(doc_df)
        print("---------------------------------------------------------------------------------------------------------")

        # get_false_negatives(doc).to_excel(doc+'_false_negatives.xlsx', encoding='utf-8')


    print("ALL DATA\n")

    print_stuff(all_df)

    # print false negatives.
    all_df = all_df[all_df['label'].isin([0.0,1.0,1,0,0,2,2.0])] 
    all_df = all_df.apply(regsearch, axis=1) 
    print('../output/'+args.regex_list.split('/')[-1]+'_errors_and_corrects.xlsx')
    with pd.ExcelWriter('../output/'+args.regex_list.split('/')[-1]+'_errors_and_corrects_on_ALL-DOCS.xlsx', engine='xlsxwriter') as fw:
        all_df[(all_df["keyword_pred"] == 0) & (all_df["label"] == 1)].to_excel(fw, sheet_name='false_negatives')
        all_df[(all_df["keyword_pred"] == 1) & (all_df["label"] == 0)].to_excel(fw, sheet_name='false_positives')
        all_df[(all_df["keyword_pred"] == 0) & (all_df["label"] == 0)].to_excel(fw, sheet_name='true_negatives')
        all_df[(all_df["keyword_pred"] == 1) & (all_df["label"] == 1)].to_excel(fw, sheet_name='true_positives')

if __name__ == "__main__":
    main()
