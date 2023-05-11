gpu_i=$1
dset="ag_news"
gpt2="gpt-j-6b"
split="dev"
bs=50

if [ ${dset} == "glue-sst2" ]
then
    maxlen=128
elif [ ${dset} == "subj" ]
then
    maxlen=160
elif [ ${dset} == "scicite" ]
then
    maxlen=240
elif [ ${dset} == "ag_news" ]
then
    maxlen=256
elif [ ${dset} == "boolq" ]
then
    maxlen=360
elif [ ${dset} == "glue-mnli" ]
then
    maxlen=160
else
    echo "Max Seq Length Not Defined!"
    exit
fi

CUDA_VISIBLE_DEVICES=${gpu_i} python eval_k1.py --dataset ${dset} --gpt2 ${gpt2} --test_batch_size ${bs} --k 1 --split ${split} --max_length_per_example ${maxlen} --use_demonstrations
