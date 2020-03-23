
CFG_PATH='/root/config/mesh2audioparams_train.yml'
#
#   --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
docker run --rm -it -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
  --gpus all --shm-size 8G \
  -v `pwd`:/root \
  -it pytorch3d-img \
  python /root/train_concert.py --config-yml ${CFG_PATH} 

