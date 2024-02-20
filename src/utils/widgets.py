import ipywidgets as widgets
from ipyfilechooser import FileChooser
from IPython.display import display


__all__ = ['make_task_widget', 'make_dataset_widget', 'make_checkpoint_file_search_widget']


def make_task_widget():
    """
    Generate an ipywidget for selecting the task from a predefined set
    """
    w = widgets.ToggleButtons(
        options=['semantic', 'panoptic'],
        description='Choose your segmentation task:',
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
        description="Choose your dataset:",
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
    w.default_path = '/'
    w.default_filename = 'checkpoint.ckpt'
    w.reset()
    
    # Shorthand reset
    w.reset(path='', filename='')
    
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
    w.title = 'Browse a relevant checkpoint file *.ckpt (you can typically use our pretrained models for this).'
    
    # Sample callback function
    def change_title(chooser):
        chooser.title = 'Selected checkpoint:'
    
    # Register callback function
    w.register_callback(change_title)

    return w
