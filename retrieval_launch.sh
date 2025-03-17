source /home/ma-user/modelarts/work/jjw/Search-R1/retriever_env.sh
file_path=/home/ma-user/modelarts/work/jjw/Search-R1/cache
index_file=/home/ma-user/modelarts/work/jjw/Search-R1/scripts/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever=/home/ma-user/modelarts/work/jjw/Search-R1/model/intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever
