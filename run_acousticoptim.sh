MESH='sphere' #/root/mesh_data/only_quad_sphere.obj'
#
SIL_DEFORM_FLAG=True
SIL_REF=/root/square_sil.png
#
ACOUST_DEFORM_FLAG=True
WHICH_ACOUST_PARAMS='2660,24920,1.25,2066,17.7,57.7,0.54,0.31,9.37,1.55'
#
OUT_NAME='test_sil_deform.obj'
#
CFG_PATH='/root/config/mesh2audioparams_train.yml'
#
docker run --rm -it -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
  --gpus all --shm-size 8G \
  -v `pwd`:/root \
  -it pytorch3d-img \
  python /root/obj_optim_acousticparams.py \
  --which_starting_mesh ${MESH} \
  --mesh_multisilhouette_optim ${SIL_DEFORM_FLAG} \
  --mesh_acousticparam_optim ${ACOUST_DEFORM_FLAG} \
  --silhouette_img_ref ${SIL_REF} \
  --which_acoustic_params ${WHICH_ACOUST_PARAMS} \
  --output_filename ${OUT_NAME} \
  --config_path ${CFG_PATH}

