import json
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from utils.utils import load_data
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification,Trainer, TrainingArguments
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils.qa_retriever import Retriever
from utils.qa_reranker import Reranker
from collections import defaultdict
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
def preprocess_faq(key_to_source_dict):
    full_context={}
    for key,source in key_to_source_dict.items():
        full_context[key]=""
        for s in source:
            full_context[key]+= s["question"]
            for ans in s["answers"]:
                full_context[key]+= ans
    return full_context
def split_chunk(page_content,max_len=256,overlap_len=100):
    cleaned_chunks = []
    i = 0
    # 暴力将整个pdf当做一个字符串，然后按照固定大小的滑动窗口切割
    all_str = ''.join(page_content)
    all_str = all_str.replace('\n', '')
    while i < len(all_str):
        cur_s = all_str[i:i+max_len]
        if len(cur_s) > 10:
            cleaned_chunks.append(cur_s)
        i += (max_len - overlap_len)

    return cleaned_chunks
if __name__ == "__main__":
    # 使用argparse解析命令列參數
    
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--task',type=str,default=["base","only_chinese","pos_rank","baai_1.5","multilingual","multilingual_bm25","summary"])
    parser.add_argument('--baai_path',type=str,default="BAAI/bge-large-zh")
    parser.add_argument('--reranker',type=str,default="BAAI/bge-reranker-large")
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epoches',type=int,default=100)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--partition',type=int,default=1,help="divide question data into several partition")
    parser.add_argument('--pid',type=int,default=0, required=True,help="which sub_question part used in program")
    parser.add_argument('--has_ground_truth',action='store_true',default=False)
    parser.add_argument('--gpu',type=int,default=0)
    args = parser.parse_args()  # 解析參數
    answer_dict = {"answers": []}  # 初始化字典
    print(args.pid,args.gpu,flush=True)
    multi_path="intfloat/multilingual-e5-large" if args.task == "multilingual" or args.task == "multilingual_bm25"  else None
    if(not os.path.exists(args.output_path)):
        os.mkdir(args.output_path)
    with open(os.path.join(args.question_path,"questions_example.json"), 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案
    if(args.has_ground_truth):
        with open(os.path.join(args.question_path,"ground_truths_example.json"), 'rb') as f:
            gt_ref = json.load(f)
        gt=gt_ref["ground_truths"]

    if args.task == "base":
        insurance_json_path="insurance.json"
        finance_json_path="finance.json"
    elif args.task == "summary":
        insurance_json_path="insurance_v2.json"
        finance_json_path="finance_summary.json"
    else:
        insurance_json_path="insurance_v2.json"
        finance_json_path="finance_v2.json"
    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance,insurance_json_path)

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance,finance_json_path)
    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
    corpus_dict_insurance={int(id):context for id,context in corpus_dict_insurance.items()}
    corpus_dict_finance={int(id):context for id,context in corpus_dict_finance.items()}

    
    #samples,labels=preprocess_sample(qs_ref["questions"],gt_ref["ground_truths"],category_data)
    faq_corpus=preprocess_faq(key_to_source_dict)

    for id,corpus in faq_corpus.items():
        faq_corpus[id]=split_chunk(corpus,max_len=256,overlap_len=100)
    for id,corpus in corpus_dict_insurance.items():
        corpus_dict_insurance[id]=split_chunk(corpus,max_len=256,overlap_len=100)
    for id,corpus in corpus_dict_finance.items():
        corpus_dict_finance[id]=split_chunk(corpus,max_len=256,overlap_len=100)
    category_corpus={
        "insurance":corpus_dict_insurance,
        "finance":corpus_dict_finance,
        "faq":faq_corpus
    }

    

    ## seperate question dataset into serveral partition and used specific part in program
    qs=qs_ref["questions"]
    
    category_qs=defaultdict(list)
    current_qs=[]
    for q in qs:
        category_qs[q["category"]].append(q)
    for cgy,q in category_qs.items():
        each_partitions_elemnts= len(q) // args.partition
        partition_q= q[args.pid *each_partitions_elemnts: (args.pid + 1) *each_partitions_elemnts]
        current_qs.extend(partition_q)

    
    correct=0
    error_answer=[]
    truth_answer=[]
    total_answer={"answers":[]}
    error_cgy={
        "faq":0,
        "insurance":0,
        "finance":0
    }
    for i,item in tqdm(enumerate(current_qs)):
        q_id=item["qid"]
        
        s_id=item["source"]
        q=item["query"]
        cgy=item["category"]
        
        corpus=category_corpus[cgy]
        
        source_ans={id:corpus[id] for id in corpus.keys() if id in s_id}
        retriever = Retriever(baai_path="BAAI/bge-reranker-large",multi_path=multi_path, corpus=source_ans,device=f"cuda:{args.gpu}")
        reranker = Reranker(rerank_model_name_or_path=args.reranker,task=args.task,device=f"cuda:{args.gpu}")
        retrieve_ans= retriever.retrieval(q)
        rerank_res_ids,rerank_scores = reranker.rerank(retrieve_ans, q, k=1)
        predicted_id=rerank_res_ids[0]
        
        output_info={
            "qid":q_id,
            "retrieve":predicted_id,
        }
        total_answer["answers"].append(output_info)
        if(args.has_ground_truth):
            gt_res_id= next((item["retrieve"] for item in gt if item["qid"]==q_id),None)
            test_info={
                "qid":q_id,
                "query": q,
                "category":cgy,
                "retrieve":gt_res_id,
                "predicted":predicted_id,
                "source":str(s_id),
                "rank_ids":str(rerank_res_ids),
                "rank_scores":str(rerank_scores)
            }
            if(predicted_id==gt_res_id):
                correct+=1
                truth_answer.append(test_info)
            else:
                error_answer.append(test_info)
                error_cgy[cgy]+=1
            print(f"Current correct:{correct}/{i+1}",flush=True)
            print(f"Current accuracy:{correct/(i+1)}",flush=True)
        

    
    output_folder=os.path.join(args.output_path,args.task)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_folder=os.path.join(args.output_path,args.task,str(args.pid)) 
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(os.path.join(args.output_path,output_folder,"pred_retrieve.json"), 'w', encoding='utf8') as f:
        json.dump(total_answer, f, ensure_ascii=False,indent=4)
    if(args.has_ground_truth):
        with open(os.path.join(args.output_path,output_folder,"correct_retrieve.json"),'w', encoding='utf8') as f:
            json.dump(truth_answer, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
        with open(os.path.join(args.output_path,output_folder,"error_retrieve.json"), 'w', encoding='utf8') as f:
            json.dump(error_answer, f, ensure_ascii=False,indent=4)
        print(f"Precision: {correct/len(gt)}",flush=True)
        print(f"Each category error:{str(error_cgy)}",flush=True)
        