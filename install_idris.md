# Création d'un environnement Pytorch

A partir de notre installation officielle : https://glcp.idris.fr/assitance-idris/recettes_installations_jz/-/blob/proposition/recettes_conda/pytorch/pytorch1.13.0_py3.10.8.md (les libs audio et nlp sont inutiles)

Liste des installations :

* Python >= 3.8
* Cuda >= 10.2
* via conda 
  * [x] pip nb_conda_kernels -c anaconda boost -c omnia eigen3 eigen -c r libiconv
* via pip
  * [x] torchvision
  * [x] torch-geometric>=2.1.0
  * [x] torch-scatter
  * [x] torch-sparse
  * [x] torch-cluster
  * [x] torch-spline-conv
  * [x] pytorch-lightning>=1.7
  * [x] hydra-core
  * [x] hydra-colorlog
  * [x] hydra-submitit-launcher
  * [x] pyrootutils
  * [x] plyfile
  * [x] h5py
  * [x] colorhash
  * [x] numba
  * [x] rich
  * torch_tb_profiler
  * [x] wandb
  * [x] gdown
  * [x] matplotlib
  * [x] seaborn
  * [x] plotly>=5.9.0 
* dependencies requiring compilation
  * [x] https://github.com/lxxue/FRNN.git src/partition/FRNN
  * [x] https://github.com/drprojects/point_geometric_features.git
  * [x] https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git src/partition/parallel_cut_pursuit
  * [x] https://gitlab.com/1a7r0ch3/grid-graph.git src/partition/grid_graph

Les librairies utilisant le GPU seront recompilées avec nos modules CUDA, cuDNN, etc... ce qui évitera par la suite des incompatibilités entre libs. 

On prend la procédure utilisée pour le module **Pytorch 1.13** adaptée aux bibliothèques listées plus haut. Ce qui donne :

| pytorch | python | numpy  |
|---------|--------|--------|
| 1.12.1  | 3.10.4 | 1.23.1 |

> à noter qu'il faut éviter d'écraser les libs dont dépend pytorch pour sa compilation (ou de réinstaller via pip si l'installation d'origine est via conda, et vice versa).

## Setup 

> ATTENTION: ne pas utiliser **$SCRATCH** quand on n'est pas dans l'équipe SU IDRIS --> ne pas oublier d'adapter $INSTALL_DIR !!

Si installation avec compilation, etc., il est conseillé de réserver un noeud sur "compil" ou "prepost" pour ne pas charger les frontales :

```sh
srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --hint=nomultithread --time=04:00:00 --partition=prepost --account $IDRPROJ@v100 bash

export INSTALL_DIR=$SCRATCH/installs/ign  # à mettre à jour
mkdir $INSTALL_DIR
```

> la partition **prepost** possède un GPU, ce qui peut être utile pour les compilations (attention il ne s'agit pas de lancer des calculs long dessus !) et a accès à internet (utile pour récupérer des paquets).

Pour faire le nettoyage :

```sh
module load anaconda-py3/2022.05
conda deactivate
export INSTALL_DIR=$SCRATCH/installs/ign  # à mettre à jour
export PYTHONUSERBASE=$INSTALL_DIR/.local
export CONDA_ENVS_PATH=$INSTALL_DIR/.conda
conda remove -y --name pytorch1.13-ign --all
rm -rfI $PYTHONUSERBASE      # /!\ uniquement si seul env dans ce repertoire
rm -rfI $CONDA_ENVS_PATH     # /!\ uniquement si seul env dans ce repertoire
```

## Creation de l'environnement

``` sh

# utiliser les modules compilés pour Intel et AMD (permet d'utiliser le module sur les A100 de Jean Zay)
module load cpuarch/amd

module load cuda/11.2 nccl/2.9.6-1-cuda cudnn/8.1.1.33-cuda  
module load gcc/8.4.1 openmpi/4.1.1-cuda intel-mkl/2020.4 magma/2.5.4-cuda  
module load cmake/3.21.3 git/2.31.1

eval "$(/gpfslocalsup/pub/anaconda-py3/2022.05/bin/conda shell.bash hook)"
conda deactivate

export INSTALL_DIR=$SCRATCH/installs/ign  # à mettre à jour
cd $INSTALL_DIR
export PYTHONUSERBASE=$INSTALL_DIR/.local
export CONDA_ENVS_PATH=$INSTALL_DIR/.conda
export PATH=$PYTHONUSERBASE/bin:$PATH

conda create -y -n pytorch1.13-ign python=3.10.8  
conda activate pytorch1.13-ign

conda install -y astunparse numpy==1.23.3 ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -y -c conda-forge tqdm libsndfile pandas xarray hdf5 protobuf
pip install --user --no-cache-dir scipy scikit-learn scikit-image simpleitk
conda config --env --add disallowed_packages 'numpy<1.23.3'
conda config --env --add disallowed_packages 'numpy>1.23.3'

mv $CONDA_PREFIX/compiler_compat/ld $CONDA_PREFIX/compiler_compat/ld.save # pour être sur d'utiliser la commande du système
```
puis :

```sh
conda install nb_conda_kernels
conda install -c anaconda boost 
conda install -c omnia eigen3 
conda install eigen 
conda install -c r libiconv 
```

## Pytorch et compagnie

### Setup

Ceci permet de compiler en utilisant nos installations locales (cuda, nccl, cudnn, etc) :

```sh
export MKL_DIRS=${MKLROOT}/lib/intel64_lin:${MKLROOT}/include  
export CMAKE_PREFIX_PATH=${MKL_DIRS}:${CMAKE_PREFIX_PATH}  
export CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath=/gpfslocalsys/cuda/11.2/lib64 -L/gpfslocalsys/cuda/11.2/lib64" 
export FORCE_CUDA=1  
export BLAS=MKL  
export USE_CUDNN=ON  
export USE_SYSTEM_NCCL=ON   
export CUDNN_LIB_DIR=/gpfslocalsup/spack_soft/cudnn/8.1.1.33-11.2/gcc-8.3.1-dgsfwc7e23vzc36jopc5nxyunptm2ieh//lib64  
export CUDNN_INCLUDE_DIR=/gpfslocalsup/spack_soft/cudnn/8.1.1.33-11.2/gcc-8.3.1-dgsfwc7e23vzc36jopc5nxyunptm2ieh//include  
export CUDNN_LIBRARY=/gpfslocalsup/spack_soft/cudnn/8.1.1.33-11.2/gcc-8.3.1-dgsfwc7e23vzc36jopc5nxyunptm2ieh//lib64/libcudnn.so  
export NCCL_ROOT=/gpfslocalsup/spack_soft/nccl/2.9.6-1/gcc-8.3.1-srwvuxi6a5sn7abu5sc3iusdirkw2cfn  
export NCCL_LIB_DIR=/gpfslocalsup/spack_soft/nccl/2.9.6-1/gcc-8.3.1-srwvuxi6a5sn7abu5sc3iusdirkw2cfn/lib  
export NCCL_INCLUDE_DIR=/gpfslocalsup/spack_soft/nccl/2.9.6-1/gcc-8.3.1-srwvuxi6a5sn7abu5sc3iusdirkw2cfn/include  
export TORCH_CUDA_ARCH_LIST="7.0;8.0"

```

### Pytorch

```sh
git clone -b v1.13.0 --single-branch --recursive https://github.com/pytorch/pytorch.git pytorch-1.13.0
cd pytorch-1.13.0/
```

On corrige un [bug](https://github.com/pytorch/kineto/pull/674/commits/97b52f1ff3ab27b52340f73415cda660fc291b83#diff-366cd2a049d8ef3616b79fed008c1a87dde06af198fede5b5492bde68406c667à) qui empêche d'utiliser tensorboard :

Dans `third_party/kineto/libkineto/src/ActivityType.cpp`, on remplace 'kernel" par "Kernel" :

```c++
    {"gpu_memcpy", ActivityType::GPU_MEMCPY},
    {"gpu_memset", ActivityType::GPU_MEMSET},
    {"Kernel", ActivityType::CONCURRENT_KERNEL}, // Legacy tools using capitalized Kernel
    {"external_correlation", ActivityType::EXTERNAL_CORRELATION},
    {"cuda_runtime", ActivityType::CUDA_RUNTIME},
```

Attention, la compilation de pytorch est très longue (1h30) et doit se faire sur **prepost**.


```sh
export PYTORCH_BUILD_VERSION=1.13.0
export PYTORCH_BUILD_NUMBER=0
time python setup.py install 2>&1 | tee install.log

real    73m27,876s
user    511m33,322s
sys     77m5,740s

```

## Torchvision

On garde le setup de **pytorch** avec les définitions de variables d'environnement :

```sh
cd $INSTALL_DIR
git clone -b v0.14.0 --single-branch --recursive https://github.com/pytorch/vision torchvision-v0.14.0
cd torchvision-v0.14.0
python setup.py install  2>&1 | tee install.log

pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_sparse.git@bc262e311f582c0591642490d6e71bc4ab636b7b # 35min

pip install --no-cache-dir torch-scatter     # 15min 
pip install --no-cache-dir torch-geometric   # 1min --> jinja2 3.1.2 (changement de version, à voir si problématique)
pip install --no-cache-dir torch-cluster     # 30min
pip install --no-cache-dir torch-spline-conv # 5min
```

## Jupyter et Tensorboard (facultatif, voire à éviter)

L'installation est problématique, en particulier pour (jupyter) Tensorboard. Nous allons proposer **Jupyter-hub** aux utilisateurs de Jean-Zay qui permettra de ne pas nécessiter une installations de ces librairies dans l'environnement conda (annonce pour la fin de l'année).

```sh
export MODULEPATH=$MODULEPATH:/gpfslocalsup/pub/modules-idris-env4/modulefiles/linux-rhel8-skylake_avx512
module load node-js/12.18.4  
module load npm/6.14.12

conda install -c conda-forge jupyter jupyterlab ipywidgets jupyter_contrib_core jupyter_nbextensions_configurator nb_conda_kernels

The following packages will be REMOVED:

  libboost-1.73.0-h28710b8_12
  py-boost-1.73.0-py310h00e6091_12

The following packages will be UPDATED:

  boost              anaconda::boost-1.73.0-py310h06a4308_~ --> conda-forge::boost-1.80.0-py310hc4a4660_4
  icu                         anaconda::icu-58.2-he6710b0_3 --> conda-forge::icu-70.1-h27087fc_0

jupyter labextension install @jupyterlab/toc @jupyter-widgets/jupyterlab-manager @jupyterlab/debugger


pip install --user --no-cache-dir tensorboard

Successfully installed absl-py-1.3.0 cachetools-5.2.0 google-auth-2.15.0 google-auth-oauthlib-0.4.6 grpcio-1.51.1 markdown-3.4.1 oauthlib-3.2.2 protobuf-3.20.3 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-oauthlib-1.3.1 rsa-4.9 tensorboard-2.11.0 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 werkzeug-2.2.2



pip install --user --no-cache-dir git+https://github.com/cliffwoolley/jupyter_tensorboard.git

Installing collected packages: webcolors, uri-template, rfc3986-validator, rfc3339-validator, jsonpointer, fqdn, arrow, isoduration, jupyter-tensorboard
Successfully installed arrow-1.2.3 fqdn-1.5.1 isoduration-20.11.0 jsonpointer-2.3 jupyter-tensorboard-0.2.0 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 uri-template-1.2.0 webcolors-1.12

> Attention à protobuf
```

C'est là que cela coince :

```sh
pip install --user --no-cache-dir git+https://github.com/chaoleili/jupyterlab_tensorboard.git # --> plantage

      subprocess.CalledProcessError: Command '['/tmp/pip-build-env-pwq3ky85/overlay/bin/jlpm', 'install']' returned non-zero exit status 1.
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for jupyterlab-tensorboard
```

## Profilers

```sh
pip install --user --no-cache-dir scalene  guppy3 py-spy 

``` 

> pas faits car erreurs : nvidia-dlprof[pytorch] torch_tb_profiler

## Librairies restantes 


Pour voir ce qui n'est pas encore installé :

```sh
conda list | egrep -e "hydra|pyrootutils|plyfile|h5py|colorhash|numba|rich|tb_profiler|wandb|gdown|seaborn|plotly"
pip list | egrep -e "hydra|pyrootutils|plyfile|h5py|colorhash|numba|rich|tb_profiler|wandb|gdown|seaborn|plotly|lightning"

```

```sh
conda install -c conda-forge plyfile h5py colorhash gdown
pip install --user --no-cache-dir hydra-core hydra-colorlog hydra-submitit-launcher 
pip install --user --no-cache-dir pytorch-lightning seaborn plotly wandb  # protobuf (3.20.x)

> Successfully installed Click-8.1.3 GitPython-3.1.29 aiohttp-3.8.3 aiosignal-1.3.1 async-timeout-4.0.2 contourpy-1.0.6 cycler-0.11.0 docker-pycreds-0.4.0 fonttools-4.38.0 frozenlist-1.3.3 fsspec-2022.11.0 gitdb-4.0.10 kiwisolver-1.4.4 lightning-utilities-0.4.2 matplotlib-3.6.2 multidict-6.0.3 pathtools-0.1.2 plotly-5.11.0 promise-2.3 protobuf-3.20.1 pytorch-lightning-1.8.4.post0 seaborn-0.12.1 sentry-sdk-1.11.1 setproctitle-1.3.2 shortuuid-1.0.11 smmap-5.0.0 tenacity-8.1.0 tensorboardX-2.5.1 torchmetrics-0.11.0 wandb-0.13.6 yarl-1.8.2


pip install --user --no-cache-dir pyrootutils 
pip install --user --no-cache-dir numba 

> Successfully installed llvmlite-0.39.1 numba-0.56.4
```

**Protobuf** est une librairies avec des versions problématiques (c'est surtout vérifiable avec Tensorflow), à voir si elle est utilisées par vos codes.

## FRNN

> https://github.com/lxxue/FRNN.git

A partir des consignes du git : 

```sh
git clone --recursive https://github.com/lxxue/FRNN.git
# install a prefix_sum routine first
cd FRNN/external/prefix_sum
pip install --user --no-cache-dir .

# install FRNN
cd ../../ # back to the {FRNN} directory
# this might take a while since I instantiate all combinations of D and K
pip install install --user --no-cache-dir .
```

## Point utils

On elève les références à Numpy et on ajoute la variable $PYTHON. 

> https://github.com/drprojects/point_geometric_features.git

**Pour la commande cmake, utiliser les fichiers modifiés dans $WORK/projects/cmake_files_for_spt.**

Vérification des paquets nécessaires :

```sh

conda list | egrep -e "boost|eigen|libiconv"
boost                     1.80.0          py310hc4a4660_4    conda-forge
boost-cpp                 1.80.0               h75c5d50_0    conda-forge
eigen                     3.4.0                h4bd325d_0    conda-forge
eigen3                    3.3.7                         0    omnia
libiconv                  1.17                 h166bdaf_0    conda-forge
```

Commandes exécutées :

```sh

git clone https://github.com/drprojects/point_geometric_features.git

cd point_geometric_features/src

export PYTHON="3.10" # AJOUT

# Copier les fichiers modifiés localement par le support IDRIS pour cmake
cd $WORK/projects/superpoint_transformer/src/partition/utils
cp $WORK/projects/cmake_files_for_spt/* ./

ln -s $CONDA_PREFIX/lib/python$PYTHON/site-packages/numpy/core/include/numpy $CONDA_PREFIX/include/numpy

cmake . -DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython$PYTHON.so -DPYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python$PYTHON -DBOOST_INCLUDEDIR=$CONDA_PREFIX/include -DEIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3

-- Found PythonLibs: /gpfsscratch/idris/sos/ssos021/installs/ign/.conda/pytorch1.13-ign/lib/libpython.so (found version "3.10.8") 
-- Found Boost: /gpfsscratch/idris/sos/ssos021/installs/ign/.conda/pytorch1.13-ign/lib/cmake/Boost-1.80.0/BoostConfig.cmake (found suitable version "1.80.0", minimum required is "1.65.0") found components: graph 
-- Found Boost: /gpfsscratch/idris/sos/ssos021/installs/ign/.conda/pytorch1.13-ign/lib/cmake/Boost-1.80.0/BoostConfig.cmake (found suitable version "1.80.0", minimum required is "1.67.0") found components: numpy310 
Boost includes ARE /gpfsscratch/idris/sos/ssos021/installs/ign/.conda/pytorch1.13-ign/include
Boost LIBRARIES ARE /gpfsscratch/idris/sos/ssos021/installs/ign/.conda/pytorch1.13-ign/lib
PYTHON LIBRARIES ARE /gpfsscratch/idris/sos/ssos021/installs/ign/.conda/pytorch1.13-ign/lib/libpython.so
-- Configuring done
-- Generating done
-- Build files have been written to: /gpfsscratch/idris/sos/ssos021/installs/ign/point_geometric_features/src


make

cd ../../..

```

## Parallel cut pursuit

```sh

git clone https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git

cd parallel-cut-pursuit/python/

python setup.py build_ext 
```

## grid-graph


```sh
git clone  https://gitlab.com/1a7r0ch3/grid-graph.git 

cd grid-graph/python/ 

python setup.py build_ext 
```

## Test sur un noeud GPU

```sh
 srun --pty --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=10 --hint=nomultithread   --time=01:00:00 --qos=qos_gpu-dev --account $IDRPROJ@gpu bash

module load cpuarch/amd

module load cuda/11.2 nccl/2.9.6-1-cuda cudnn/8.1.1.33-cuda  
module load gcc/8.4.1 openmpi/4.1.1-cuda intel-mkl/2020.4 magma/2.5.4-cuda  
module load cmake/3.21.3 git/2.31.1

eval "$(/gpfslocalsup/pub/anaconda-py3/2022.05/bin/conda shell.bash hook)"
conda deactivate

export INSTALL_DIR=$SCRATCH/installs/ign  # à mettre à jour
cd $INSTALL_DIR
export PYTHONUSERBASE=$INSTALL_DIR/.local
export CONDA_ENVS_PATH=$INSTALL_DIR/.conda
export PATH=$PYTHONUSERBASE/bin:$PATH
conda activate pytorch1.13-ign


cd $INSTALL_DIR/point_geometric_features/
python demo.py  # ok

cd $INSTALL_DIR/parallel-cut-pursuit/python/
python example_tomography.py  # ok


cd $INSTALL_DIR/parallel-cut-pursuit/python/
python test.py # ok

```