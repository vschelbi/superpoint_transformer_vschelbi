import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display


__all__ = [
    'make_experiment_widgets', 'make_task_widget', 'make_dataset_widget', 
    'make_checkpoint_file_search_widget']


# Hardcoded list of released experiment configs
EXPERIMENT_CONFIGS = {
    'semantic': [
        'dales',
        'dales_11g',
        'dales_nano',
        'kitti360',
        'kitti360_11g',
        'kitti360_nano',
        's3dis',
        's3dis_11g',
        's3dis_nano',
        's3dis_room',
        'scannet',
        'scannet_11g',
        'scannet_nano'],
    'panoptic': [
        'dales',
        'dales_11g',
        'dales_nano',
        'kitti360',
        'kitti360_11g',
        'kitti360_nano',
        's3dis',
        's3dis_11g',
        's3dis_11g_with_stuff',
        's3dis_nano',
        's3dis_nano_with_stuff',
        's3dis_room',
        's3dis_with_stuff',
        'scannet',
        'scannet_11g',
        'scannet_nano']}


def make_experiment_widgets():
    """
    Generate two co-dependent ipywidgets for selecting the task and 
    experiment from a predefined set of experiment configs.
    """
    default_task = list(EXPERIMENT_CONFIGS.keys())[0]
    default_expe = EXPERIMENT_CONFIGS[default_task][0]
    
    w_task = widgets.ToggleButtons(
        options=EXPERIMENT_CONFIGS.keys(),
        value=default_task,
        description="👉 Choose a segmentation task:",
        disabled=False,
        button_style='')

    w_expe = widgets.ToggleButtons(
        options=EXPERIMENT_CONFIGS[default_task],
        value=default_expe,
        description="👉 Choose an experiment:",
        disabled=False,
        button_style='')

    # Define a function that updates the content of one widget based on 
    # what we selected for the other
    def update(*args):
        print(f"selected : {w_task.value}")
        w_expe.options = EXPERIMENT_CONFIGS[w_task.value]
        
    w_task.observe(update)

    display(w_task)
    display(w_expe)
    
    return w_task, w_expe



def make_task_widget():
    """
    Generate an ipywidget for selecting the task from a predefined set
    """
    w = widgets.ToggleButtons(
        options=['semantic', 'panoptic'],
        value='semantic',
        description="👉 Choose a segmentation task:",
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Semantic segmentation', 'Panoptic segmentation'])
    display(w)
    return w


def make_dataset_widget():
    """
    Generate an ipywidget for selecting the dataset from a predefined 
    set
    """
    w = widgets.ToggleButtons(
        options=['dales', 'kitti360', 's3dis', 's3disroom', 'scannet'],
        value='s3dis',
        description="👉 Choose a dataset:",
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['DALES', 'KITTI-360', 'S3DIS', 'S3DIS room-by-room', 'ScanNet'])
    display(w)
    return w


def make_checkpoint_file_search_widget():
    """
    Generate an ipywidget for locally browsing a checkpoint file
    """
    # Create and display a FileChooser widget
    w = FileChooser('', layout = widgets.Layout(width='80%'))
    display(w)
    
    # Change defaults and reset the dialog
    w.default_path = '..'
    w.default_filename = ''
    w.reset()
    
    # Shorthand reset
    w.reset(path='..', filename='')
    
    # Restrict navigation to /Users
    w.sandbox_path = '/'
    
    # Change hidden files
    w.show_hidden = False
    
    # Customize dir icon
    w.dir_icon = '/'
    w.dir_icon_append = True
    
    # Switch to folder-only mode
    w.show_only_dirs = False
    
    # Set a file filter pattern (uses https://docs.python.org/3/library/fnmatch.html)
    # w.filter_pattern = '*.txt'
    w.filter_pattern = '*.ckpt'
    
    # Set multiple file filter patterns (uses https://docs.python.org/3/library/fnmatch.html)
    # w.filter_pattern = ['*.jpg', '*.png']
    
    # Change the title (use '' to hide)
    w.title = "👉 Choose a checkpoint file *.ckpt relevant to your experiment (eg use our or your own pretrained models for this):"
    
    # Sample callback function
    def change_title(chooser):
        chooser.title = 'Selected checkpoint:'
    
    # Register callback function
    w.register_callback(change_title)

    return w
