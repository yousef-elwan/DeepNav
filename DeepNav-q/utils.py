"""
Utility functions for file/directory management and checkpoint retrieval.
"""
import os
import shutil
import time
import tensorflow as tf

def create_trial_tree(trial_number, session_mode):
    """
    Creates a structured directory tree for a given trial.
    Archives old trial folders in 'Override' mode instead of deleting them.
    """
    base_results_dir = "DeepNav_results"
    trial_root_folder = os.path.join(base_results_dir, f"trial_{trial_number:03d}")

    if session_mode == "Override" and os.path.exists(trial_root_folder):
        print(f"Override mode: Archiving existing folder for trial {trial_number}...")
        archive_dir = os.path.join(base_results_dir, "_archive")
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        archived_name = f"{os.path.basename(trial_root_folder)}_{timestamp}"
        try:
            shutil.move(trial_root_folder, os.path.join(archive_dir, archived_name))
            print(f"Archived to: {os.path.join(archive_dir, archived_name)}")
        except Exception as e:
            print(f"Could not archive folder: {e}")

    trial_tree = {
        "trial_root_folder": trial_root_folder,
        "weights_folder": os.path.join(trial_root_folder, "weights"),
        "history_csv_file": os.path.join(trial_root_folder, "model_history_log.csv"),
    }

    if session_mode != "Resume":
        folders_to_create = [
            os.path.join(trial_root_folder, "plots", "best"),
            os.path.join(trial_root_folder, "plots", "worst"),
            os.path.join(trial_root_folder, "plots", "other"),
            trial_tree["weights_folder"],
        ]
        for folder in folders_to_create:
            os.makedirs(folder, exist_ok=True)

    print(f"\n*** Trial results will be saved in: {trial_root_folder} ***")
    return trial_tree

def retrieve_latest_weights(weights_folder):
    """
    Finds the latest TensorFlow checkpoint in a directory.
    Returns the epoch number and the full path to the latest checkpoint file.
    """
    latest_checkpoint = tf.train.latest_checkpoint(weights_folder)
    if not latest_checkpoint:
        return 0, None
    try:
        filename = os.path.basename(latest_checkpoint)
        last_epoch = int(filename.split('-')[0].split('.')[1])
    except (ValueError, IndexError):
        print(f"Warning: Could not parse epoch from checkpoint: {latest_checkpoint}")
        last_epoch = 0
    return last_epoch, latest_checkpoint