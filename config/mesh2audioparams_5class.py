RANDOM_SEED: 0

PHASE: "training"
EXPERIMENT_NAME: "concerthalloptim"
RESULTS_DIR: "/storage/mesh2acoustic_training_results"
OVERFIT: True

SHAPENET_DATA:
    PATH: "/storage/concert_dataset"

OPTIM:
    BATCH_SIZE: 4
    VAL_BATCH_SIZE: 4
    WORKERS: 8
    EPOCH: 300
    LR: 0.0002
    MOMENTUM: 0.9
    WEIGHT_DECAY: 0.005
    CLIP_GRADIENTS: 12.5

GCC:
    INPUT_MESH_FEATS: 3
    HIDDEN_DIMS: [128, 256, 512]
    CLASSES: 5
    CONV_INIT: "normal"
