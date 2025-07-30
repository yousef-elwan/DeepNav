import os
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import pickle
import glob
import json

# --- Quaternion Math Helper Functions ---
def quaternion_conjugate(q):
    """Calculates the conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q0):
    """Multiplies two quaternions."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([
        w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
        w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
        w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
        w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0
    ])

def create_dataset(session_data, column_names):
    """
    Creates or loads a windowed dataset for training and evaluation.
    This version uses the quaternion CHANGE (delta) as the label.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, os.path.pardir, "DeepNav_data")
    sets_subdirs = ["training", "validation"]

    print("Creating new dataset for delta orientation prediction...")
    csvs_root_directory = os.path.join(data_directory, "combined_csvs", "trimmed")
    
    combined_windowed_features = {}
    combined_windowed_labels = {}
    flights_dictionaries = {"training": {}, "validation": {}}

    for set_subdir in sets_subdirs:
        print(f"Processing '{set_subdir}' set...")
        csvs_directory = os.path.join(csvs_root_directory, set_subdir)
        x_list, y_list = [], []

        for flight_file in sorted(os.listdir(csvs_directory)):
            csv_path = os.path.join(csvs_directory, flight_file)
            df = pd.read_csv(csv_path)

            features = df[column_names["features"]].to_numpy()[1:, :]
            features_diff_col = df[column_names["features_diff"]].to_numpy()
            absolute_quaternions = df[column_names["labels"]].to_numpy()

            features_diff = np.diff(features_diff_col, axis=0)
            processed_features = np.hstack((features, features_diff))

            windowed_features = []
            windowed_delta_labels = [] # Labels are now delta quaternions
            window_size = session_data["window_size"]

            for i in range(len(processed_features) - window_size + 1):
                feature_window = processed_features[i : i + window_size]

                # The label is the quaternion change over the window
                # q_end = delta_q * q_start  =>  delta_q = q_end * conjugate(q_start)
                q_start = absolute_quaternions[i + window_size - 2]
                q_end = absolute_quaternions[i + window_size - 1]
                
                delta_q_label = quaternion_multiply(q_end, quaternion_conjugate(q_start))
                
                windowed_features.append(feature_window)
                windowed_delta_labels.append(delta_q_label)

            x_one_flight = np.array(windowed_features)
            y_one_flight = np.array(windowed_delta_labels)

            # For evaluation, we need features and the FULL ground truth absolute sequence
            ground_truth_for_eval = absolute_quaternions[window_size-1:]
            flights_dictionaries[set_subdir][flight_file[:-4]] = (x_one_flight, ground_truth_for_eval)
            
            x_list.append(x_one_flight)
            y_list.append(y_one_flight)

        combined_windowed_features[set_subdir] = np.vstack(x_list)
        combined_windowed_labels[set_subdir] = np.vstack(y_list)
        
        # Shuffle the combined dataset
        p = np.random.permutation(len(combined_windowed_features[set_subdir]))
        combined_windowed_features[set_subdir] = combined_windowed_features[set_subdir][p]
        combined_windowed_labels[set_subdir] = combined_windowed_labels[set_subdir][p]

    print("\n--- Dataset Shapes (Delta Orientation) ---")
    for set_subdir in sets_subdirs:
        print(f"Shape of '{set_subdir}' features: {combined_windowed_features[set_subdir].shape}")
        print(f"Shape of '{set_subdir}' labels:   {combined_windowed_labels[set_subdir].shape}")
        print("----")
    
    training_dataset = tf.data.Dataset.from_tensor_slices((
        combined_windowed_features["training"],
        combined_windowed_labels["training"]
    ))
    validation_dataset = tf.data.Dataset.from_tensor_slices((
        combined_windowed_features["validation"],
        combined_windowed_labels["validation"]
    ))
    
    return training_dataset, validation_dataset, flights_dictionaries["training"], flights_dictionaries["validation"]