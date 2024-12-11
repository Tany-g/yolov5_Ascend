#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="yolov5s_v7.0"

cur_path=`pwd`
model_name=yolov5s
batch_size=128

for para in $*
do
   if [[ $para == --model_name* ]];then
      	model_name=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   fi
done

# 校验是否指定了device_id,分动态分配device_id与手动指定device_id,此处不需要修改
local_rank=0
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
elif [ ${local_rank} ];then
    export ASCEND_DEVICE_ID=${local_rank}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/
else
    mkdir -p ${cur_path}/test/output/$ASCEND_DEVICE_ID/
fi

source ${cur_path}/test/env_npu.sh

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

python3 val.py \
    --weights yolov5s.pt \
    --data coco.yaml \
    --img 640 \
    --iou 0.65 \
    --half \
    --device 0 \
    --batch-size $batch_size > ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_eval_1p.log 2>&1 &

wait

# #训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#最后一个迭代FPS值
acc=`grep -a 'IoU=0.50:0.95' ${cur_path}/test/output/$ASCEND_DEVICE_ID/train_eval_1p.log|grep 'Average Precision'|awk 'NR==1'| awk -F " " '{print $13}'`

#打印，不需要修改
echo "Final Train Accuracy : $acc"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log