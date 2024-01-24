export CUDA_VISIBLE_DEVICES="0"
cuda='0'
data_dir="NERD"
output_dir="results/${data_dir}/text-kl_beit3"
save_dir="./model/model_${data_dir}"
dataset="./data/${data_dir}"
lr=0.00002
batch_size=1
batch_size=32
num_sen_per_entity=128
epoch=1
pretrain_epoch=1
pretrained_model="epoch_${pretrain_epoch}.pkl"

output="${output_dir}_${seed}_(beit3)"
save_path="${save_dir}_${seed}"
echo "---pretrain and makedist---"
python -u main_beit3.py -cuda=$cuda -output=$output -save_path=$save_path -dataset=$dataset -pretrained_model=$pretrained_model -pretrain -make_dist -lr=$lr -epoch=$epoch -num_sen_per_entity=$num_sen_per_entity -batch_size=$batch_size
echo "---expand---"
python -u main_beit3.py -cuda=$cuda -output=$output -save_path=$save_path -dataset=$dataset -pretrained_model=$pretrained_model -test -mode='multi2' -lr=$lr -epoch=$epoch -num_sen_per_entity=$num_sen_per_entity -batch_size=$batch_size
python -u main_beit3.py -cuda=$cuda -output=$output -save_path=$save_path -dataset=$dataset -pretrained_model=$pretrained_model -test -mode='text' -lr=$lr -epoch=$epoch -num_sen_per_entity=$num_sen_per_entity -batch_size=$batch_size
python -u main_beit3.py -cuda=$cuda -output=$output -save_path=$save_path -dataset=$dataset -pretrained_model=$pretrained_model -test -mode='visual' -lr=$lr -epoch=$epoch -num_sen_per_entity=$num_sen_per_entity -batch_size=$batch_size