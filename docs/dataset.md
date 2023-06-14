# Datasets

All datasets inherit from the `torch_geometric` `Dataset` class, allowing for 
automated download (when allowed), preprocessing and inference-time transforms. 
See the [official documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)
for more details. 

## `data/` structure 
<details>
<summary><b>Data directory structure.</b></summary>

Datasets are stored under the following structure:

```
└── data
    ├── dales                                         # Structure for DALES
    │   ├── DALESObjects.tar.gz                         # (optional) Downloaded zipped dataset
    │   ├── raw                                         # Raw dataset files
    │   │   └── {{train, test}}                           # DALES' split/tile.ply structure
    │   │       └── {{tile_name}}.ply
    │   └── processed                                   # Preprocessed data
    |       └── {{train, val, test}}                      # Dataset splits
    |           └── {{preprocessing_hash}}                  # Preprocessing folder
    │               └── {{tile_name}}.h5                      # Preprocessed tile file
    │    
    ├── kitti360                                      # Structure for KITTI-360
    │   ├── raw                                         # Raw dataset files
    │   │   ├── data_3d_semantics_test.zip              # (optional) Downloaded zipped test dataset
    │   │   ├── data_3d_semantics.zip                   # (optional) Downloaded zipped train dataset
    │   │   └── data_3d_semantics                       # Contains all raw train and test sequences
    │   │       └── {{sequence_name}}                     # KITTI-360's sequence/static/window.ply structure
    │   │           └── static
    │   │               └── {{window_name}}.ply
    │   └── processed                                   # Preprocessed data
    │       └── {{train, val, test}}                      # Dataset splits
    │           └── {{preprocessing_hash}}                  # Preprocessing folder
    │               └── {{sequence_name}}
    │                   └── {{window_name}}.h5                # Preprocessed window file
    │    
    └── s3dis                                         # Structure for S3DIS
        ├── Stanford3dDataset_v1.2.zip                  # (optional) Downloaded zipped dataset
        ├── raw                                         # Raw dataset files
        │   └── Area_{{1, 2, 3, 4, 5, 6}}                 # S3DIS's area/room/room.txt structure
        │       └── {{room_name}}  
        │           └── {{room_name}}.txt
        └── processed                                   # Preprocessed data
            └── {{train, val, test}}                      # Dataset splits
                └── {{preprocessing_hash}}                  # Preprocessing folder
                    └── Area_{{1, 2, 3, 4, 5, 6}}.h5          # Preprocessed Area file
```
</details>

💾❓**Already have the dataset on your machine** ? Save memory by simply 
symlinking or copying the files to `data/<dataset_name>/raw/`, following the 
above structure.

## Automatic download
Following `torch_geometric`'s `Dataset` behaviour:
- missing files in `data/<dataset_name>/raw` structure ➡ automatic download
- missing files in `data/<dataset_name>/processed` structure ➡ automatic preprocessing

However, some datasets (_e.g._ KITTI-360) require you to *manually download**
from their official webpage. For those, you will need to manually setup the 
above-described `data/` structure.

## Pre-transforms, transforms, on-device transforms

Pre-transforms are the functions making up the preprocessing. 
These are called only once and their output is saved in 
`data/<dataset_name>/processed/`. These typically encompass neighbor search and 
partition construction.

The transforms are called by the `Dataloaders` at batch-creation time. These 
typically encompass sampling and data augmentations and are performed on CPU, 
before moving the batch to the GPU.

On-device transforms, are transforms to be performed on GPU. These are 
typically compute intensive operations that could not be done once and for all 
at preprocessing time, and are too slow to be performed on CPU.

## Preprocessing hash
Different from `torch_geometric`, you can have **multiple 
preprocessed versions** of each dataset, identified by their preprocessing hash.

This hash will change whenever the preprocessing configuration 
(_i.e._ pre-transforms) is modified in an impactful way (_e.g._ changing the 
partition regularization). 

Modifications of the transforms and on-device 
transforms will not affect your preprocessing hash.
