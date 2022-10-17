#!/bin/bash

# Recover the project directory from the position of the install.sh script
HERE=`dirname $0`
HERE=`realpath $HERE`


# Local variables
PROJECT_NAME=spt
YML_FILE=${HERE}/${PROJECT_NAME}.yml
PYTHON=3.8
TORCH=1.12.0
CUDA_SUPPORTED=(10.2)


# Installation script for Anaconda3 environments
echo "#############################################"
echo "#                                           #" 
echo "#           Deep View Aggregation           #"
echo "#                 Installer                 #"
echo "#                                           #" 
echo "#############################################"
echo
echo


echo "_______________ Prerequisites _______________"
echo "  - conda"
echo "  - cuda >= 10.2 (tested with `echo ${CUDA_SUPPORTED[*]}`)"
echo "  - gcc >= 7"
echo
echo


echo "____________ Pick conda install _____________"
echo
# Recover the path to conda on your machine
CONDA_DIR=`realpath ~/anaconda3`

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide you conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh
echo
echo


echo "_____________ Pick CUDA version _____________"
echo

CUDA_VERSION=`nvcc --version | grep release | sed 's/.* release //' | sed 's/, .*//'`

# If CUDA version not supported, ask whether to proceed
if [[ ! " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_VERSION} " ]]
then
    echo "Found CUDA ${CUDA_VERSION} installed, is not among tested versions: "`echo ${CUDA_SUPPORTED[*]}`
    echo "This may cause downstream errors when installing PyTorch and PyTorch Geometric dependencies, which you might solve by manually modifying setting the wheels in this script."
    read -p "Proceed anyways ? [y/n] " -n 1 -r; echo
    if !(test -z $REPLY) && [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi


echo "________________ Installation _______________"
echo

# Create deep_view_aggregation environment from yml
conda env create -f ${YML_FILE}  #********************************* CREATE YML FILE
#conda create --name $PROJECT_NAME python=$PYTHON -y

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${PROJECT_NAME}


#*********************************

conda install pip nb_conda_kernels -y
pip install matplotlib
pip install plotly==5.9.0
pip install "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash
pip install "notebook>=5.3" "ipywidgets>=7.5"
pip install ipykernel
pip install torch==1.12.0 torchvision
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install plyfile
pip install h5py
pip install colorhash
pip install seaborn
pip3 install numba --user
pip install pytorch-lightning --user

#*********************************

echo
echo "___________________ FRNN ___________________"
echo
git clone --recursive https://github.com/lxxue/FRNN.git superpoint_transformer/partition/FRNN

# install a prefix_sum routine first
cd superpoint_transformer/partition/FRNN/external/prefix_sum
python setup.py install

# install FRNN
cd ../../ # back to the {FRNN} directory
python setup.py install
cd ../../../


echo
echo "________________ Point Utils _______________"
echo
conda install -c anaconda boost -y
conda install -c omnia eigen3 -y
conda install eigen -y
conda install -c r libiconv -y
ln -s $CONDA_PREFIX/lib/python$PYTHON/site-packages/numpy/core/include/numpy $CONDA_PREFIX/include/numpy
cd superpoint_transformer/partition/utils
cmake . -DPYTHON_LIBRARY=$CONDA_PREFIX/lib/libpython$PYTHON.so -DPYTHON_INCLUDE_DIR=$CONDA_PREFIX/include/python$PYTHON -DBOOST_INCLUDEDIR=$CONDA_PREFIX/include -DEIGEN3_INCLUDE_DIR=$CONDA_PREFIX/include/eigen3
make
cd ../../..


echo
echo "________________ Cut-Pursuit _______________"
echo

# Clone parallel-cut-pursuit and grid-graph repos
git clone https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git superpoint_transformer/partition/parallel_cut_pursuit
git clone https://gitlab.com/1a7r0ch3/grid-graph.git superpoint_transformer/partition/grid_graph

# Compile the projects
python scripts/setup_dependencies.py build_ext
