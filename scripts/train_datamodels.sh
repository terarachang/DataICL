gpu_i=$1
dset="scicite"
gpt2="gpt-j-6b"
logits_dir="Dicl/${gpt2}/label_${dset}"
dm_dir="out_datamodel/${gpt2}"


if [ ${dset} == "glue-sst2" ]
then
    patterns=16
elif [ ${dset} == "boolq" ]
then
    patterns=16
elif [ ${dset} == "subj" ]
then
    patterns=16
elif [ ${dset} == "ag_news" ]
then
    patterns=24
elif [ ${dset} == "glue-mnli" ]
then
    patterns=6
elif [ ${dset} == "scicite" ]
then
    patterns=6
else
    echo "Task Not Defined!"
    exit
fi

mode="Datamodels"
echo ${mode}

# on CPU
# train datamodels with all the data
CUDA_VISIBLE_DEVICES=-1 python train_datamodels.py --task ${dset} --ckpt_dir ${logits_dir} --datamodel_dir ${dm_dir} --feat_type feat1

# fine-tune datamodels based on label patterns
for (( i = 0; i < ${patterns}; i++ ))
do
    echo pattern $i
    CUDA_VISIBLE_DEVICES=-1 python train_datamodels.py --do_init --task ${dset} --ckpt_dir ${logits_dir} --datamodel_dir ${dm_dir} --feat_type feat1-$i
done

