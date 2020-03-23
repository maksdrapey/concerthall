MESH='plane' #/root/mesh_data/only_quad_sphere.obj'
OUT_NAME='dreaming_results_gconv012'
CFG_PATH='/root/config/style_transfer_train.yml'
#
#   --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
docker run --rm -it -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
  --gpus all --shm-size 8G \
  -v `pwd`:/root \
  -it pytorch3d-img \
  python /root/deepdream_loop.py \
  --which_starting_mesh ${MESH} \
  --output_dir ${OUT_NAME} \
  --config_path ${CFG_PATH} 