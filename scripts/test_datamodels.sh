gpu_i=$1
dset="scicite"
gpt2="gpt-j-6b"
ckpt_dir="Dicl/${gpt2}/test_${dset}"
dm_dir="out_datamodel/${gpt2}"


if [ ${dset} == "glue-sst2" ]
then
    if [ ${gpt2} == "gpt-j-6b" ]
    then
        n_subsets_arr=("50000")
    else
        n_subsets_arr=("25000")
    fi
    patterns=16
elif [ ${dset} == "boolq" ]
then
    if [ ${gpt2} == "gpt-j-6b" ]
    then
        n_subsets_arr=("50000")
    else
        n_subsets_arr=("25000")
    fi
    patterns=16
elif [ ${dset} == "subj" ]
then
    n_subsets_arr=("50000")
    patterns=16
elif [ ${dset} == "ag_news" ]
then
    patterns=24
elif [ ${dset} == "scicite" ]
then
    n_subsets_arr=("20000")
    patterns=6
else
    echo "Task Not Defined!"
    exit
fi

for n in ${n_subsets_arr[@]}
do

    # test datamodels based on label patterns
    for (( i = 0; i < ${patterns}; i++ ))
    do
        echo ${dm_dir}
        CUDA_VISIBLE_DEVICES=-1 python test_datamodels.py --task ${dset} --ckpt_dir ${ckpt_dir} --datamodel_dir ${dm_dir}/${dset}/$n-feat1-$i --feat_type feat1-$i
    done

    python report_test_datamodels.py --task ${dset} --model ${gpt2} --datamodel_dir ${dm_dir} --n_patterns ${patterns} --n_train_sets $n

done

