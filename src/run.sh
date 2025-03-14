

mkdir -p "log/multilingual_bm25"
for pid in {0..9}
do
    gpu=$((pid % 4))
    logfile="log/multilingual_bm25/$pid.log"
    nohup python RAG.py \
        --question_path ../dataset/preliminary \
        --source_path ../reference \
        --output_path ../output \
        --pid $pid \
        --partition 10 \
        --task "multilingual_bm25" \
        --gpu $gpu \
        > "$logfile" 2>&1 &
        #--baai_path BAAI/bge-large-zh-v1.5 \
        #--reranker BAAI/bge-reranker-v2-m3
    echo "Started process with pid=$pid, log file: $logfile"
done