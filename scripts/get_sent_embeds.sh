gpu_i=$1
task_arr=("glue-sst2" "boolq" "subj" "scicite" "ag_news")
plm='sbert'

for t in ${task_arr[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_i} python analysis/get_sent_embed.py --task ${t} --plm ${plm}
done
