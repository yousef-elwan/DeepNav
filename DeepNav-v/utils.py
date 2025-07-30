import os
import shutil
import re

def create_trial_tree(project_root, trial_number, session_mode):
    """
    Creates the directory structure for a training trial based on the project root.
    """
    trial_root_folder = os.path.join(project_root, "DeepNav_results", f"trial_{trial_number:03d}")

    trial_tree = {
        "trial_root_folder": trial_root_folder,
        "weights_folder": os.path.join(trial_root_folder, "weights"),
        "history_csv_file": os.path.join(trial_root_folder, "model_history_log.csv")
    }

    if session_mode in ["Resume", "Evaluate"]:
        os.makedirs(trial_tree["weights_folder"], exist_ok=True)
        return trial_tree
    
    if session_mode == "Override" and os.path.exists(trial_root_folder):
        shutil.rmtree(trial_root_folder)

    os.makedirs(trial_tree["weights_folder"], exist_ok=True)
    for set_type in ["training", "validation"]:
        for data_type in ["differenced", "reconstructed"]:
            for folder_type in ["best", "worst", "other"]:
                os.makedirs(os.path.join(trial_root_folder, set_type, data_type, folder_type), exist_ok=True)

    print(f"\n*** Created/Cleared Trial Folder: {trial_root_folder} ***")
    return trial_tree


def get_epoch_from_filename(filename):
    """Extracts the epoch number from a checkpoint filename using regex."""
    if not filename: return 0
    match = re.search(r'ep\.(\d+)', filename)
    return int(match.group(1)) if match else 0

def retrieve_latest_weights(weights_folder):
    """
    Finds the checkpoint file with the highest epoch number.
    Returns the file path and the epoch number.
    """
    if not os.path.exists(weights_folder) or not os.listdir(weights_folder):
        return None, 0

    saved_weights_files = [f for f in os.listdir(weights_folder) if f.endswith('.weights.h5')]
    if not saved_weights_files:
        return None, 0

    latest_file = max(saved_weights_files, key=get_epoch_from_filename)
    last_epoch = get_epoch_from_filename(latest_file)
    
    return os.path.join(weights_folder, latest_file), last_epoch