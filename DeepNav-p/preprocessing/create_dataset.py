import os
import numpy as np
import pandas as pd
import tensorflow as tf

def create_dataset(session_data, column_names):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, os.path.pardir, "DeepNav_data")
    sets_subdirs = ["training", "validation"]
    csvs_root_directory = os.path.join(data_directory, "combined_csvs", "trimmed")
    
    # --- KEY CHANGE: Define a scaling factor ---
    LABEL_SCALING_FACTOR = 10.0
    print(f"Applying label scaling with factor: {LABEL_SCALING_FACTOR}")
    # -----------------------------------------
    
    combined_features = {"training": [], "validation": []}
    combined_labels = {"training": [], "validation": []}
    flights_dictionaries = {"training": {}, "validation": {}}

    for set_subdir in sets_subdirs:
        print(f"Processing '{set_subdir}' set...")
        csvs_directory = os.path.join(csvs_root_directory, set_subdir)
        if not os.path.exists(csvs_directory):
            print(f"Warning: Directory not found, skipping: {csvs_directory}")
            continue

        for flight_file in sorted(os.listdir(csvs_directory)):
            csv_path = os.path.join(csvs_directory, flight_file)
            df = pd.read_csv(csv_path)

            features = df[column_names["features"]].to_numpy()
            absolute_positions = df[column_names["labels"]].to_numpy()

            window_size = session_data["window_size"]
            if len(features) <= window_size: continue

            windowed_features, windowed_delta_labels = [], []
            for i in range(len(features) - window_size):
                windowed_features.append(features[i : i + window_size])
                delta_p_label = (absolute_positions[i + window_size] - absolute_positions[i + window_size - 1])
                
                # --- KEY CHANGE: Scale the labels ---
                windowed_delta_labels.append(delta_p_label * LABEL_SCALING_FACTOR)
                # ------------------------------------

            x_one_flight = np.array(windowed_features)
            y_one_flight = np.array(windowed_delta_labels)

            ground_truth_for_eval = absolute_positions[window_size:]
            flights_dictionaries[set_subdir][flight_file.replace('.csv', '')] = (x_one_flight, ground_truth_for_eval)
            
            combined_features[set_subdir].append(x_one_flight)
            combined_labels[set_subdir].append(y_one_flight)

    final_features, final_labels = {}, {}
    for set_subdir in sets_subdirs:
        if not combined_features[set_subdir]:
            final_features[set_subdir] = np.zeros((0, session_data["window_size"], session_data["n_features"]))
            final_labels[set_subdir] = np.zeros((0, 3))
            continue
            
        final_features[set_subdir] = np.vstack(combined_features[set_subdir])
        final_labels[set_subdir] = np.vstack(combined_labels[set_subdir])
        
        p = np.random.permutation(len(final_features[set_subdir]))
        final_features[set_subdir] = final_features[set_subdir][p]
        final_labels[set_subdir] = final_labels[set_subdir][p]

    print("\n--- Dataset Shapes ---")
    for s in sets_subdirs:
        print(f"'{s}' features: {final_features[s].shape}")
        print(f"'{s}' labels:   {final_labels[s].shape}")
    
    train_ds = tf.data.Dataset.from_tensor_slices((final_features["training"], final_labels["training"]))
    val_ds = tf.data.Dataset.from_tensor_slices((final_features["validation"], final_labels["validation"]))
    
    return train_ds, val_ds, flights_dictionaries["training"], flights_dictionaries["validation"]