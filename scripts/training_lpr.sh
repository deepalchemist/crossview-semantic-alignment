# Please refer to project, dataset, model yaml
# If you do not have root privileges, then disabling the peer to peer transport
# NCCL_P2P_DISABLE=1 or NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=PIX 

#############
### DEBUG ###
#############
#CUDA_VISIBLE_DEVICES=0 python mmf_cli/run.py config=projects/videotoshop/configs/e2e_pretraining_vic_vim_rec.yaml model=rice dataset=videotoshop


#############################
### SINGLE NODES TRAINING ###
#############################
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --master_addr=172.17.175.111 --master_port=29501 mmf_cli/run.py config=projects/videotoshop/configs/e2e_pretraining_vic.yaml model=rice dataset=videotoshop

############################
### MULTI NODES TRAINING ###
############################
#NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=172.17.175.111 --master_port=29501 mmf_cli/run.py config=projects/videotoshop/configs/e2e_pretraining_vic_vim_rec.yaml model=rice dataset=videotoshop

#NCCL_DEBUG=INFO python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=172.17.175.111 --master_port=29501 mmf_cli/run.py config=projects/videotoshop/configs/e2e_pretraining_vic_vim_rec.yaml model=rice dataset=videotoshop
