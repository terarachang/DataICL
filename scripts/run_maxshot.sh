gpu_i=$1
gpt2="gpt-j-6b"
k=4
split="test"
task_arr=("glue-sst2" "boolq" "ag_news" "subj" "scicite")
n_trunc=50
usize=20



run_icl() {
    echo ${mode}
    CUDA_VISIBLE_DEVICES=${gpu_i} python evaluate.py --dataset $t --gpt2 ${gpt2} --mode ${mode} --k $k --split ${split} --test_batch_size ${bs} --max_length_per_example ${maxlen} --use_demonstrations
}


for t in ${task_arr[@]}
do
    if [ $t == "glue-sst2" ]
    then
        maxlen=80
        k=24
        bs=12
    elif [ $t == "subj" ]
    then
        maxlen=80
        k=24
        bs=12
    elif [ $t == "boolq" ]
    then
        maxlen=250
        k=8
        bs=10
    elif [ $t == "ag_news" ]
    then
        maxlen=120
        k=16
        bs=12
    elif [ $t == "scicite" ]
    then
        maxlen=160
        k=12
        bs=12
    else
        echo "Max Seq Length Not Defined!"
        exit
    fi

    mode="MaxShot"
    python select_maxshots.py --n_shots $k --model ${gpt2} --task $t
    run_icl
    
done

