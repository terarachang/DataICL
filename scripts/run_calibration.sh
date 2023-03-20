gpu_i=$1
gpt2="gpt-j-6b"
k=4
split="test"
task_arr=("glue-sst2" "boolq" "subj" "scicite" "ag_news")
n_trunc=50
usize=20



run_icl() {
    echo ${mode}
    CUDA_VISIBLE_DEVICES=${gpu_i} python calib_evaluate.py --dataset $t --gpt2 ${gpt2} --mode ${mode} --k $k --split ${split} --test_batch_size ${bs} --max_length_per_example ${maxlen} --use_demonstrations
}


for t in ${task_arr[@]}
do
    if [ $t == "glue-sst2" ]
    then
        if [ ${gpt2} == "gpt-j-6b" ]
        then
            maxlen=128
        else
            maxlen=80
        fi
        bs=32
    elif [ $t == "subj" ]
    then
        maxlen=80
        bs=50
    elif [ $t == "boolq" ]
    then
        maxlen=256
        if [ ${gpt2} == "opt-13b" ]
        then
            bs=15
        else
            bs=20
        fi
    elif [ $t == "ag_news" ]
    then
        maxlen=128
        if [ ${gpt2} == "opt-13b" ]
        then
            bs=25
        else
            bs=50
        fi
    elif [ $t == "scicite" ]
    then
        maxlen=160
        bs=32
        k=3
    else
        echo "Max Seq Length Not Defined!"
        exit
    fi

    mode="All"
    run_icl

    
done

