import sys, os

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # this is for the .py script but does not work in a notebook
# file_path = os.path.dirname(os.path.abspath(''))
sys.path.append(file_path)
# sys.path.append(os.path.join(file_path, "grid-graph/python/bin"))
# sys.path.append(os.path.join(file_path, "parallel-cut-pursuit/python/wrappers"))
sys.path.append(os.path.join(file_path, "superpoint_transformer/partition/grid_graph/python/bin"))
sys.path.append(os.path.join(file_path, "superpoint_transformer/partition/parallel_cut_pursuit/python/wrappers"))

from superpoint_transformer.utils.cpu import available_cpu_count
print(f'CPUs available: {available_cpu_count()}')

#-----------------------------------------------------------------------

# DATA_ROOT
DATA_ROOT = '/media/drobert-admin/DATA2'
# DATA_ROOT = '/var/data/drobert'
