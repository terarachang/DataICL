gpu_i=$1
seg_i=$2
p_i=$3
dset="glue-sst2"
gpt2="gpt-j-6b"
k=4

if [ ${dset} == "glue-sst2" ]
then
    if [ ${gpt2} == "gpt-j-6b" ]
    then
        maxlen=128
    else
        maxlen=80
    fi
    bs=32
elif [ ${dset} == "subj" ]
then
    maxlen=80
    bs=32
elif [ ${dset} == "glue-mnli" ]
then
    maxlen=100
    bs=32
    k=3
elif [ ${dset} == "scicite" ]
then
    maxlen=160
    bs=32
    k=3
elif [ ${dset} == "ag_news" ]
then
    maxlen=128
    if [ ${gpt2} == "opt-13b" ]
    then
        bs=25
    else
        bs=50
    fi
elif [ ${dset} == "boolq" ]
then
    maxlen=256
    if [ ${gpt2} == "opt-13b" ]
    then
        bs=15
    else
        bs=20
    fi
else
    echo "Max Seq Length Not Defined!"
    exit
fi

CUDA_VISIBLE_DEVICES=${gpu_i} python dicl_data_collection.py --dataset ${dset} --gpt2 ${gpt2} --test_batch_size ${bs} --use_demonstrations --k $k --permute_fn_id ${p_i} --segment_id ${seg_i} --split dev --max_length_per_example ${maxlen}
