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
        bs=32
    elif [ $t == "subj" ]
    then
        maxlen=80
        bs=50
    elif [ $t == "ag_news" ]
    then
        maxlen=128
        if [ ${gpt2} == "opt-13b" ]
        then
            bs=25
        else
            bs=50
        fi
    elif [ $t == "boolq" ]
    then
        maxlen=256
        if [ ${gpt2} == "opt-13b" ]
        then
            bs=15
        else
            bs=20
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

    ckpt=Dicl/${gpt2}/unlabel_$t

    mode="CondAcc-good-unlabeled"
    python select_condacc.py --model ${gpt2} --task $t --ckpt_dir ${ckpt} --useful_size ${usize} --n_trunc ${n_trunc} --is_unlabel
    run_icl

    top=5
    mode="TopPrompts-${top}-unlabeled"
    python baseline_top_prompts.py --model ${gpt2} --task $t --ckpt_dir ${ckpt} --n_trunc ${n_trunc} --n_top ${top} --is_unlabel
    run_icl

    mode="OneShot-unlabeled"
    python baseline_oneshot.py --model ${gpt2} --task $t --useful_size ${usize} --n_trunc ${n_trunc} --is_unlabel
    run_icl

    mode="All-unlabeled"
    run_icl
    
done

