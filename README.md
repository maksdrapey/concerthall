## concerthall optimization framework

This optimization framework is run using the Paperspace Gradient Experiment tool. The following information is the same across 
scripts provided in this repository:


The docker image/container used for this repository is:
`acarlson32/pytorch3dimg:firstimage`


The workspace is:
`https://github.com/alexacarlson/concerthall`


### Training the Graph Convolutional Neural network to map 3D meshes --> audio quality parameters

Command format:
`run_train_graphconv_regression.sh`

Example Command:
`run_train_graphconv_regression.sh`


### Running the Acoustic quality optimization using the trained graph convolutional network

Command format:

`bash run_acousticoptim.sh INPUT_MESH SILHOUETTE_DEFORM_FLAG SILHOUETTE_REF_IMG ACOUSTIC_DEFORM_FLAG WHICH_ACOUSTIC_PARAMS OUTPUT_NAME`

Example Command:

`bash run_acousticoptim.sh '/storage/quad_sphere.obj' True '/storage/square_sil.png' True '2660,24920,1.25,2066,17.7,57.7,0.54,0.31,9.37,1.55' 'test_sil_deform.obj'`
