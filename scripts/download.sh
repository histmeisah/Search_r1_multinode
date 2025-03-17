
save_path=/home/ma-user/modelarts/work/jjw/Search-R1/cache

# python download.py --save_path $save_path

# cat $save_path/part_* > e5_Flat.index

gzip -d $save_path/wiki-18.jsonl.gz
