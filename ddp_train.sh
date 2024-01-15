# DDP training
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="127.0.0.1" --master_port=25641 \
#main_task_retrieval.py --do_train --sim_header mean_pooling --one_stage --embedding_sim --output_dir ckpts/ddp_lpr4m_icl --epochs=1 --batch_size=64 --fp16 --dataset lpr4m --expand_msrvtt_sentences

# Single node training
python -m torch.distributed.launch --nproc_per_node=4 --master_port=25641 \
main_task_retrieval.py --do_train \
--epochs=1 --batch_size=128 --fp16 \
--dataset lpr4m --expand_msrvtt_sentences  \
--output_dir ckpts/gpu4_full_lpr4m_icl_pmd_rec \
--sim_header cross_attention --cross_num_hidden_layers 2 --recons_feat --embedding_sim \
#--data_root "/mnt/csip-113/yangwenjie/dataset/lpr4m/raw_data/" \
#--lr 3e-4 --max_words 32 --max_frames 10 --batch_size_val 128 \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 --linear_patch 2d \
#--pretrained_clip_name ViT-B/32 \

# (a) instance contrastive learning
#--sim_header mean_pooling --one_stage --embedding_sim \
# (b) icl+patch matching decoder 
#--sim_header cross_attention --cross_num_hidden_layers 2 --embedding_sim \
# (c) icl+pmd+rec
#--sim_header cross_attention --cross_num_hidden_layers 2 --recons_feat --embedding_sim \
# (d) icl+pmd+rec+ipd
#--sim_header cross_attention --cross_num_hidden_layers 2 --recons_feat --ipd --embedding_sim \

#--sim_header mean_pooling --loose_type --ipd --embedding_sim \
#--sim_header cross_attention --cross_num_hidden_layers 2 --add_text --embedding_sim \
#--sim_header cross_attention --cross_num_hidden_layers 2 --ipd --add_text --recons_feat --embedding_sim \
