## concerthall optimization framework

This optimization framework is run using the Paperspace Gradient Experiment tool. The following information is the same across 
scripts provided in this repository:


The docker image/container used for this repository is:
`acarlson32/pytorch3dimg:firstimage`


The workspace is:
`https://github.com/alexacarlson/concerthall`


### Training the Graph Convolutional Neural network to map 3D meshes --> audio quality parameters
To train your own graph convolutional network, you first need to create a folder in paperspace storage at `/storage/concert_dataset` upload your data there. All of your meshes should be `.obj` files and should exist in a subdirectory of the dataset: `/storage/concert_dataset/OBJdatabase/`. You will need to have a csv file of acoustic paramters (where each line is in the format `mesh.obj acoust_param1 acoust_param2 ... acoust_param10`) located at and named `/storage/concert_dataset/AcousticParameters.csv`.

Command format:
`bash run_train_graphconv_regression.sh`

Example Command:
`bash run_train_graphconv_regression.sh`

The results (network weights and training loss values) will be saved to `/storage/mesh2acoustic_training_results`.

### Running the Acoustic quality optimization using the trained graph convolutional network

Command format:

`bash run_acousticoptim.sh \
      INPUT_MESH \
      SILHOUETTE_DEFORM_FLAG \
      SILHOUETTE_REF_IMG \
      ACOUSTIC_DEFORM_FLAG \
      WHICH_ACOUSTIC_PARAMS \
      OUTPUT_NAME`
      
      
where `INPUT_MESH` is the absolute filepath of the mesh you would like to deform, `SILHOUETTE_DEFORM_FLAG` is either True or False depending on if you would like to deform the mesh to optimize for a specific 2D silhouette, `SILHOUETTE_REF_IMG` is the absolute filepath of the silhouette image you would like to use for mesh deformation, `ACOUSTIC_DEFORM_FLAG` is either True or False depending on if you would like to deform the mesh to optimize for acostic quality, `WHICH_ACOUSTIC_PARAMS` is a string of the 10 acoustic quality parameters (each parameter needs to be separated by a comma with no spaces in between, see example below), and `OUTPUT_NAME` is the absolute filepath where you would like the deformed output mesh to be saved.


Example Command:

`bash run_acousticoptim.sh \
    '/storage/quad_sphere.obj' \
    True \
    '/storage/square_sil.png' \
    True \
    '2660,24920,1.25,2066,17.7,57.7,0.54,0.31,9.37,1.55' \
    'test_sil_deform.obj'`
    
    
