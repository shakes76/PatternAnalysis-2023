import os
import shutil

def reset_dir(folder):
    """
        reset folder: remove all the files in this folder. If this folder is not exist, then it'll make new one.
    """
    
    # Try to remove folder
    try :
        shutil.rmtree(folder)
    except Exception as e:
        # This folder is not exist.
        pass
    
    os.mkdir(folder)