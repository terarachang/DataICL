gpu_i=$1
gpt2="gpt-j-6b"
k=4
split="test"
task_arr=("imdb" "contrast_boolq")
n_trunc=50
usize=20



run_icl() {
    echo ${mode}
    CUDA_VISIBLE_DEVICES=${gpu_i} python evaluate.py --dataset $t --gpt2 ${gpt2} --mode ${mode} --k $k --split ${split} --test_batch_size ${bs} --max_length_per_example ${maxlen} --use_demonstrations --trunc_method 'middle' --source_task ${src}
}


for t in ${task_arr[@]}
do
    if [ $t == "imdb" ]
    then
        maxlen=160
        bs=25
        src='glue-sst2' 
    elif [ $t == "contrast_boolq" ]
    then
        maxlen=256
        bs=15
        src='boolq' 
    else
        echo "Max Seq Length Not Defined!"
        exit
    fi

    mode="CondAcc-good"
    run_icl

    mode="Datamodels"
    run_icl

    mode="All"
    run_icl

    mode="TopPrompts"
    run_icl

    
done

