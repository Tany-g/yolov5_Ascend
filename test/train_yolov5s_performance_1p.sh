#!/bin/bash

#网络名称,同目录名称,需要模型审视修改
Network="yolov5s_ID4103_for_PyTorch_v7.0"

cur_path=`pwd`
model_name=yolov5s
batch_size=64

for para in $*
do
   if [[ $para == --model_name* ]];then
      	model_name=`echo ${para#*=}`
   elif [[ $para == --batch_size* ]];then
      	batch_size=`echo ${para#*=}`
   elif [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
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


#创建DeviceID输出目录，不需要修改
if [ -d ${cur_path}/test/output/${ASCEND_DEVICE_ID} ];
	then
	   rm -rf ${cur_path}/test/output/${ASCEND_DEVICE_ID}
		mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
	else
	   mkdir -p ${cur_path}/test/output/${ASCEND_DEVICE_ID}
	fi

# 非平台场景时source 环境变量
check_etp_flag=$(env | grep etp_running_flag)
etp_flag=$(echo ${check_etp_flag#*=})
if [ x"${etp_flag}" != x"true" ]; then
    source ${test_path_dir}/env_npu.sh
else
    sed -i "s|./coco|$data_path|g" ./data/coco.yaml
fi


#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

python3 -u train.py \
    --data ./data/coco.yaml \
    --cfg yolov5s.yaml \
    --weights '' \
    --optimizer 'NpuFusedSGD' \
    --batch-size $batch_size \
    --device $ASCEND_DEVICE_ID \
    --epochs 1 > ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}_performance_1p.log 2>&1 &

wait

#训练结束时间，不需要修改
end_time=$(date +%s)
echo "end_time: ${end_time}"
e2e_time=$(( $end_time - $start_time ))

#最后一个迭代FPS值
FPS=`grep -a 'FPS'  ${cur_path}/test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}_performance_1p.log|awk 'END {print}'| awk -F "FPS:" '{print $2}' | awk -F "]" '{print $1}'`

#打印，不需要修改
echo "Final Performance images/sec : $FPS"
echo "E2E Training Duration sec : $e2e_time"

#稳定性精度看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'perf'

##获取性能数据，不需要修改
#单迭代训练时长
TrainingTime=`awk 'BEGIN{printf "%.2f\n", '${batch_size}'*1000/'${FPS}'}'

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" > $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "ActualFPS = ${FPS}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >> $cur_path/test/output/$ASCEND_DEVICE_ID/${CaseName}.log
rm -rf $data_path/labels/*.cache
