#!/bin/bash

# 集群训练命令样例
# bash test/train_yolov5s_cluster.sh --nnodes=nodes_number --node_rank=node_id --master_addr=x.x.x.x --master_port=50000 --batch_size=1024 --epochs=1

#网络名称,同目录名称,需要模型审视修改
Network="yolov5s_for_PyTorch_v7.0"

cur_path=`pwd`
model_name=yolov5s
batch_size=512
device_number=8
#精度训练epoch 数设置,默认300
epochs=300
head_rank=0
for para in $*
do
    if [[ $para == --model_name* ]];then
        model_name=`echo ${para#*=}`
    elif [[ $para == --batch_size* ]];then
        batch_size=`echo ${para#*=}`
    elif [[ $para == --epochs* ]];then
        epochs=`echo ${para#*=}`
    elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --node_rank* ]];then
        node_rank=`echo ${para#*=}`
    elif [[ $para == --master_addr* ]];then
        master_addr=`echo ${para#*=}`
    elif [[ $para == --master_port* ]];then
        master_port=`echo ${para#*=}`
    elif [[ $para == --nnodes* ]];then
        nnodes=`echo ${para#*=}`
    elif [[ $para == --device_number* ]];then
        device_number=`echo ${para#*=}`
    elif [[ $para == --head_rank* ]];then
        head_rank=`echo ${para#*=}`
    fi
done

###############指定训练脚本执行路径###############
# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
     test_path_dir=${cur_path}
     cd ..
     cur_path=$(pwd)
else
     test_path_dir=${cur_path}/test
fi

# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
     source ${test_path_dir}/env_npu.sh
fi

export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I |awk '{print $1}')

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

export MASTER_ADDR=${master_addr}
export MASTER_PORT=${master_port}
export WORLD_SIZE=$((nnodes * device_number))

for i in $(seq ${head_rank} $((head_rank + device_number - 1)))
do
    if [ -d ${cur_path}/test/output/${i} ];
    then
        rm -rf ${cur_path}/test/output/${i}
        mkdir -p ${cur_path}/test/output/${i}
    else
        mkdir -p ${cur_path}/test/output/${i}
    fi
    rank=$((i + node_rank * device_number))
    export RANK=${rank}
    export LOCAL_RANK=$i

    if [ $(uname -m) = "aarch64" ]
    then
        let p_start=0+24*i
        let p_end=23+24*i
        taskset -c $p_start-$p_end python3 -u train.py \
              --data coco.yaml \
              --cfg yolov5s.yaml \
              --addr ${master_addr} \
              --port ${master_port} \
              --weights '' \
              --batch-size $batch_size \
              --local_rank $i \
              --optimizer 'NpuFusedSGD' \
              --device_num 8 \
              --epochs ${epochs} > $cur_path/test/output/${i}/train_${WORLD_SIZE}p_${i}.log 2>&1 &
    
    else
        python3 -u train.py \
              --data coco.yaml \
              --addr ${master_addr} \
              --port ${master_port} \
              --cfg yolov5s.yaml \
              --weights '' \
              --batch-size $batch_size \
              --local_rank $i \
              --optimizer 'NpuFusedSGD' \
              --device_num 8 \
              --epochs ${epochs} > ${cur_path}/test/output/${i}/train_${WORLD_SIZE}p_${i}.log 2>&1 &
    
    fi                        
done

wait


if [[ x"${master_addr}" == x"${HCCL_IF_IP}" ]];then

    # #训练结束时间，不需要修改
    end_time=$(date +%s)
    echo "end_time: ${end_time}"
    e2e_time=$(( $end_time - $start_time ))

    FPS=`grep -a 'FPS'  ${cur_path}/test/output/${head_rank}/train_${WORLD_SIZE}p_${head_rank}.log|awk 'END {print}'| awk -F "FPS:" '{print $2}' | awk -F "]" '{print $1}'`

    #取acc值
    acc=`grep -a 'IoU=0.50:0.95' ${cur_path}/test/output/${head_rank}/train_${WORLD_SIZE}p_${head_rank}.log|grep 'Average Precision'|awk 'NR==1'| awk -F " " '{print $13}'`

    #打印，不需要修改
    echo "Final Performance images/sec : $FPS"
    echo "Final Train Accuracy : $acc"
    echo "E2E Training Duration sec : $e2e_time"

    #稳定性精度看护结果汇总
    #训练用例信息，不需要修改
    BatchSize=${batch_size}
    DeviceType=`uname -m`
    CaseName=${Network}_bs${BatchSize}_${WORLD_SIZE}'p'

    ##获取性能数据，不需要修改
    #单迭代训练时长
    TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'`

    #关键信息打印到${CaseName}.log中，不需要修改
    echo "Network = ${Network}" > $cur_path/test/output/${head_rank}/${CaseName}.log
    echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/${head_rank}/${CaseName}.log
    echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/${head_rank}/${CaseName}.log
    echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/${head_rank}/${CaseName}.log
    echo "CaseName = ${CaseName}" >> $cur_path/test/output/${head_rank}/${CaseName}.log
    echo "ActualFPS = ${FPS}" >> $cur_path/test/output/${head_rank}/${CaseName}.log
    echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/${head_rank}/${CaseName}.log
    echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/${head_rank}/${CaseName}.log
    rm -rf $data_path/labels/*.cache
fi
