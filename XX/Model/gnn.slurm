#!/bin/bash
#SBATCH -p gpu                                 # 作业提交的队列
#SBATCH -N 1                                   # 作业申请 1 个节点
#SBATCH -n 16                                  # 总计申请 8 个核心
#SBATCH --ntasks-per-node=16                   # 每节点申请 8 个核心
#SBATCH -t 120:00:00                           # 任务运行的最长时间
#SBATCH --gres=gpu:0                           # 单个节点使用 1 块 GPU 卡

source /public/home/zhaochenyang/anaconda3/etc/profile.d/conda.sh
conda activate GNN

cd /public/home/zhaochenyang/job/GNN/KACL-dgl/Model

python main.py
