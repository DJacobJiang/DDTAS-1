#!/usr/bin/env bash
DATA=cub
DATA_ROOT=DataSet
Gallery_eq_Query=True

LOSS=SoftCon

CHECKPOINTS=ckps
R=.pth.tar

if_exist_mkdir ()
{
    dirname=$1
    if [ ! -d "$dirname" ]; then
    mkdir $dirname
    fi
}

if_exist_mkdir ${CHECKPOINTS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}
if_exist_mkdir ${CHECKPOINTS}/${LOSS}/${DATA}

if_exist_mkdir result
if_exist_mkdir result/${LOSS}
if_exist_mkdir result/${LOSS}/${DATA}

NET=BN_Inception

DIM=512
ALPHA=40
LR=1e-5
BatchSize=80
ValBatchSize=10
RATIO=0.16
MARGIN=0.1
DDTAS=0
DSIZE=50
DATE=20221011


SAVE_DIR=${CHECKPOINTS}/${LOSS}/${DATA}/${DATE}/${NET}-DIM-${DIM}-lr-${LR}-BatchSize-${BatchSize}-dsize-${DSIZE}-DDTAS-${DDTAS}
if_exist_mkdir ${SAVE_DIR}

# if [ ! -n "$1" ] ;then
#echo "Begin Training!"
CUDA_VISIBLE_DEVICES=2,3 python DDTAS_train.py --net ${NET} \
--data $DATA \
--net_t ${NET} \
--data_root ${DATA_ROOT} \
--init random \
--lr $LR \
--dim $DIM \
--alpha $ALPHA \
--num_instances   5 \
--val_num_instances 2 \
--batch_size ${BatchSize} \
--val_batch_size ${ValBatchSize} \
--epoch 1000 \
--loss $LOSS \
--width 227 \
--save_dir ${SAVE_DIR} \
--save_step 5 \
--ratio ${RATIO} \
--margin ${MARGIN} \
--MMSI ${MMSI} \
--d_size ${DSIZE} \
