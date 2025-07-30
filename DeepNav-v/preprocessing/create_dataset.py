import os
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import pickle
import glob

def create_dataset(project_root, session_data, colum_names):
    """
    Creates or loads a windowed time-series dataset from flight CSVs.
    """
    data_directory = os.path.join(project_root, "DeepNav_data")
    
    all_sets_features = {}
    all_sets_labels = {}
    flights_dictionaries = {"training": {}, "validation": {}}
    sets_subdirs = ["training", "validation"]

    if session_data.get("dataset_name"):
        print("Retrieving dataset:", session_data["dataset_name"])
        datasets_directory = os.path.join(data_directory, "datasets", session_data["dataset_name"])

        if not os.path.exists(datasets_directory):
            print(f"Error: Dataset folder not found at {datasets_directory}")
            print("Please run with --new_dataset flag to create it.")
            exit()

        with open(os.path.join(datasets_directory, "features_labels.pkl"), 'rb') as f:
            data = pickle.load(f)
            all_sets_features = data["features"]
            all_sets_labels = data["labels"]

        with open(os.path.join(datasets_directory, "flights_dictionaries.pkl"), 'rb') as f:
            flights_dictionaries = pickle.load(f)

        with open(os.path.join(datasets_directory, "features_labels_names.txt"), 'r') as f:
            loaded_names = eval(f.read())
            session_data['n_features'] = len(loaded_names['features']) + len(loaded_names.get('features_diff', []))
            session_data['n_labels'] = len(loaded_names['labels'])

    else:
        csvs_root_directory = os.path.join(data_directory, "combined_csvs", "trimmed")
        
        if not os.path.exists(csvs_root_directory):
            print(f"Error: Raw CSV data folder not found at {csvs_root_directory}")
            exit()

        n_logs = len(glob.glob(os.path.join(csvs_root_directory, '**', '*.csv'), recursive=True))
        n_features = str(session_data["n_features"])
        n_labels = str(session_data["n_labels"])
        
        dataset_name = (f"T{session_data['trial_number']:03d}_logs{n_logs}"
                        f"_F{n_features}L{n_labels}_W{session_data['window_size']}"
                        f"_{datetime.datetime.now():%d%b%Y_%H%M}")

        datasets_directory = os.path.join(data_directory, "datasets", dataset_name)
        os.makedirs(datasets_directory, exist_ok=True)
        
        print("Creating new dataset:", dataset_name, "...")
        session_data["dataset_name"] = dataset_name

        for set_subdir in sets_subdirs:
            print(f"Processing '{set_subdir}' set...")
            csvs_directory = os.path.join(csvs_root_directory, set_subdir)
            x_list, y_list = [], []

            if not os.path.exists(csvs_directory):
                print(f"Warning: Directory not found for '{set_subdir}' set: {csvs_directory}")
            else:
                for flight_file in sorted(os.listdir(csvs_directory)):
                    if not flight_file.endswith('.csv'): continue
                    
                    flight_name = flight_file.replace('.csv', '')
                    csv_file_name = os.path.join(csvs_directory, flight_file)
                    
                    try:
                        features_base = pd.read_csv(csv_file_name, usecols=colum_names["features"]).to_numpy()
                        labels = pd.read_csv(csv_file_name, usecols=colum_names["labels"]).to_numpy()
                    except ValueError as e:
                        print(f"Warning: Skipping file {flight_file} due to column error: {e}")
                        continue
                    
                    labels = np.diff(labels, axis=0)

                    if colum_names.get("features_diff"):
                        features_diff = pd.read_csv(csv_file_name, usecols=colum_names["features_diff"]).to_numpy()
                        features_diff = np.diff(features_diff, axis=0)
                        features_base = features_base[1:len(labels)+1, :]
                        features = np.hstack((features_base, features_diff))
                    else:
                        features = features_base[1:len(labels)+1, :]

                    windowed_features, windowed_labels = [], []
                    for i in range(len(labels) - session_data["window_size"] + 1):
                        one_window = features[i : i + session_data["window_size"], :]
                        one_label = labels[i + session_data["window_size"] - 1, :]
                        windowed_features.append(one_window)
                        windowed_labels.append(one_label)
                    
                    if not windowed_features: continue

                    x_one_flight = np.array(windowed_features, dtype=np.float32)
                    y_one_flight = np.array(windowed_labels, dtype=np.float32)

                    flights_dictionaries[set_subdir][flight_name] = (x_one_flight, y_one_flight)
                    x_list.append(x_one_flight)
                    y_list.append(y_one_flight)
            
            # ### تم التصحيح: التعامل مع حالة عدم وجود بيانات في المجلد
            if not x_list: 
                print(f"Warning: No valid data found for '{set_subdir}' set. This set will be empty.")
                all_sets_features[set_subdir] = np.empty((0, session_data["window_size"], session_data["n_features"]), dtype=np.float32)
                all_sets_labels[set_subdir] = np.empty((0, session_data["n_labels"]), dtype=np.float32)
                continue

            all_sets_features[set_subdir] = np.vstack(x_list)
            all_sets_labels[set_subdir] = np.vstack(y_list)

            shuffled_indices = np.arange(len(all_sets_features[set_subdir]))
            np.random.shuffle(shuffled_indices)
            all_sets_features[set_subdir] = all_sets_features[set_subdir][shuffled_indices]
            all_sets_labels[set_subdir] = all_sets_labels[set_subdir][shuffled_indices]

        # Save the dataset to files
        with open(os.path.join(datasets_directory, "features_labels.pkl"), 'wb') as f:
            pickle.dump({"features": all_sets_features, "labels": all_sets_labels}, f)

        with open(os.path.join(datasets_directory, "flights_dictionaries.pkl"), 'wb') as f:
            pickle.dump(flights_dictionaries, f)

        with open(os.path.join(datasets_directory, "features_labels_names.txt"), 'w') as f:
            f.write(str(colum_names))
    
    # Defensive checks to provide clearer errors
    if "training" not in all_sets_features or "validation" not in all_sets_labels:
        raise KeyError("Failed to populate both 'training' and 'validation' keys in the dataset dictionaries.")

    training_dataset = tf.data.Dataset.from_tensor_slices((all_sets_features["training"], all_sets_labels["training"]))
    
    # ### تم التصحيح: إصلاح الترتيب الخاطئ للمعاملات هنا
    validation_dataset = tf.data.Dataset.from_tensor_slices((all_sets_features["validation"], all_sets_labels["validation"]))

    return training_dataset, validation_dataset, flights_dictionaries["training"], flights_dictionaries["validation"]