model="configs/adni_schaefer100/TUs_graph_classification_Contrasformer_adni_schaefer100_100k.json"
echo ${model}
python3 main.py --gpu_id 0 --node_feat_transform pearson --max_time 60 --config $model --init_lr 1e-4 --min_lr 1e-5 --dropout 0.5 --contrast --lambda1 1e-4 --lambda2 1.0 --lambda3 0.1 --L 2
