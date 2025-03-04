# AI CUP 2024 YuShan Artificial Intelligence Open Challenge - Application of RAG and LLM in Financial Q&A
This project is for AIcup RAG and LLM in financial Q&A contest
## Introduction
| Category | description | reference file |
| -------- | -------- | -------- |
| faq     | 玉山銀行官方網站上的常見問題    | .json     |
| insurance     | 玉山銀行代銷的保險產品之保單條款     | .pdf     |
| finance     | 公開資訊觀測站上的上市公司財務報告    | .pdf     |
||||

Dataset Structure
```
├── dataset
│ ├── preliminary
│ │ └── questions_example.json
│ │ └── ground_truths_example.json 
└── reference
    ├── faq
    │ └── pid_map_content.json 
    ├── insurance
    │   ├── 1.pdf
    │   ├── 2.pdf
    │   └── ... 
    └── finance
        ├── 0.pdf 
        ├── 1.pdf 
        └── ...
```
`questions_example.json`: example questions(150 questions)
```=
{
    "questions": [
        {
            "qid": 1,
            "source": [442, 115, 440, 196, 431, 392, 14, 51], 
            "query": "匯款銀行及中間行所收取之相關費用由誰負擔?",
            "category": "insurance"
        },
        // 後面題目省略... 
    ]
}
```
| Column | Type | description |
| -------- | -------- | -------- |
| qid    | integer   | 題號     |
| query  | string    | 問題     |
| source | list of integer    | 能夠回答問題的可能選項，數字的意義為文件編號 ( pid )，可在資料夾 reference 中找到對應的檔案或內容     |
| category  | string    | 資料類型，reference 裡有對應的資料夾存放該類型的文件 |
||||


`ground_truths_example.json`: answer for example questions
```
{
    "answers": [
        {
            "qid": 1,
            "retrieve": 926
        },
        // 後面題目省略... 
    ]
}
```


`questions_preliminary.json`: contest questions(900 questions)
`pred_retrieve_example_2.json`: answer for contest questions
## How to Run the code
### enviroments setting up 
⚠️ Please install anaconda3 before you execute this code
```
conda env create -f aicup.yml
```
### execute
```
bash run.sh
```
#### run.sh
we just divide question into serveral part and execute program concurrently in different gpu devices to accelerate training time
```bash


mkdir -p "log/[Your_log_folder]"
for pid in {0..9}
do
    gpu=$((pid % [num_of_your_gpus]))
    logfile="log/log/[Your_log_folder]/$pid.log"
    nohup python multichoice.py \
        --question_path ../dataset/preliminary \
        --source_path ../reference \
        --output_path ../output \
        --pid $pid \
        --partition 10 \
        --task "[our_model]" \
        --gpu $gpu \
        > "$logfile" 2>&1 &
        #--baai_path BAAI/bge-large-zh-v1.5 \
        #--reranker BAAI/bge-reranker-v2-m3
    echo "Started process with pid=$pid, log file: $logfile"
done
```
After execute program, we need to merge every answer part into one to evaluate
```python
python output/merge.py --folder "[log folder name your save in run.sh]"
```
## Task we used
* base: read pdf without filter symmbol
* only_chinese: filter all symmbol, only remain chinese word
* pos_rank: add position score into rerank model
* baai_1.5: BAAI newest retriever model
* multilingual: newest RAG model in huggingface
* multilingual_bm25: only multilingual and bm25 to reranker

### position ranking
because we used chunk to divide context before retriever model, we noticed that in rerank model scores, there exist repeat id with different chunk score, so we add priority position score into rerank score.
## LLM model implement
### retriever
[bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh)
[multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)
### reranker
[bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
## result
### Testing data
Precision: 0.9467 
Each category error:{'faq': 0, 'insurance': 4, 'finance': 4}

⚠️ There are 2 controversial answer in test dataset 

:::
### Contest data
![](contest_score.png)

