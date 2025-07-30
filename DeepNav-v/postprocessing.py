import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re

def flight_pdf_plots(file_name, ground_truth, predictions, signal_meta):
    """
    Creates a PDF file with plots for each signal, using metadata for labels and dt.
    """
    time_length = ground_truth.shape[0]
    dt = signal_meta.get("dt", 0.2)
    time_steps = np.arange(time_length) * dt / 60

    signal_names = signal_meta.get("names", [f"Signal {i+1}" for i in range(ground_truth.shape[1])])

    with PdfPages(file_name) as pdf:
        for signal_number, signal_name in enumerate(signal_names):
            plt.figure(figsize=(10, 6))
            true_signal = ground_truth[:, signal_number]
            predicted_signal = predictions[:, signal_number]
            signal_MAE = np.mean(np.absolute(true_signal - predicted_signal))

            plt.plot(time_steps, true_signal, label="Ground Truth (EKF)")
            plt.plot(time_steps, predicted_signal, linestyle='--', label="Prediction (Network)")
            plt.grid(True, linestyle=':')
            plt.ylabel(signal_name)
            plt.xlabel("Time (minutes)")
            plt.title(f"{signal_name}\nMAE = {signal_MAE:.3f}")
            plt.legend()
            pdf.savefig()
            plt.close()

def evaluate_all_flights(model, train_flights_dict, val_flights_dict, trial_folder, signal_meta, n_extreme_flights=10):
    """
    Evaluates the model on all flights, generates plots, and calculates error metrics.
    """
    flights_summary = {}
    set_names = ["training", "validation"]
    for flights_dict, set_name in zip([train_flights_dict, val_flights_dict], set_names):
        
        flights_list = sorted(flights_dict.items())
        if not flights_list:
            print(f"No flights to evaluate for '{set_name}' set.")
            flights_summary[set_name] = []
            continue

        total_flights = len(flights_list)
        flights_errors = {}
        set_summary = []

        for flight_number, (flight_name, one_flight_data) in enumerate(flights_list):
            print(f"Evaluating {set_name} flight {flight_number+1}/{total_flights}: {flight_name}")
            
            features, ground_truth_diff = one_flight_data
            if features.ndim == 0 or features.shape[0] == 0:
                print(f"Skipping flight {flight_name} due to empty features.")
                continue

            predictions_diff = model.predict(features)

            ground_truth_reconstructed = np.cumsum(ground_truth_diff, axis=0)
            predictions_reconstructed = np.cumsum(predictions_diff, axis=0)

            max_velocity_error = np.max(np.linalg.norm(ground_truth_reconstructed - predictions_reconstructed, axis=1))

            pdf_name = f"{flight_name}_MVE_{max_velocity_error:.2f}.pdf"
            
            pdf_name_diff = os.path.join(trial_folder, set_name, "differenced", pdf_name)
            flight_pdf_plots(pdf_name_diff, ground_truth_diff, predictions_diff, signal_meta)
            
            pdf_name_recon_other = os.path.join(trial_folder, set_name, "reconstructed", "other", pdf_name)
            flight_pdf_plots(pdf_name_recon_other, ground_truth_reconstructed, predictions_reconstructed, signal_meta)

            flights_errors[pdf_name] = max_velocity_error

            flight_duration_min = ground_truth_reconstructed.shape[0] * signal_meta.get("dt", 0.2) / 60
            flight_id_match = re.search(r'\d+', flight_name)
            flight_id = int(flight_id_match.group(0)) if flight_id_match else -1
            set_summary.append([flight_id, flight_duration_min, max_velocity_error])

        flights_summary[set_name] = set_summary
        
        sorted_flights = sorted(flights_errors.items(), key=lambda x: x[1])
        
        old_base = os.path.join(trial_folder, set_name, "reconstructed", "other")
        best_base = os.path.join(trial_folder, set_name, "reconstructed", "best")
        worst_base = os.path.join(trial_folder, set_name, "reconstructed", "worst")

        num_to_show = min(n_extreme_flights, len(sorted_flights))

        best_filenames = [sorted_flights[i][0] for i in range(num_to_show)]
        worst_filenames = [sorted_flights[-(i+1)][0] for i in range(num_to_show)]

        for fname in best_filenames:
            source_path = os.path.join(old_base, fname)
            dest_path = os.path.join(best_base, fname)
            if os.path.exists(source_path):
                os.rename(source_path, dest_path)

        for fname in worst_filenames:
            if fname not in best_filenames:
                source_path = os.path.join(old_base, fname)
                dest_path = os.path.join(worst_base, fname)
                if os.path.exists(source_path):
                    os.rename(source_path, dest_path)
            
    return flights_summary

def summarize_session(project_root, trial_tree, model, session_data, flights_summary):
    """
    Appends a summary of the training session to a main CSV file in the project root.
    """
    history_file = trial_tree["history_csv_file"]
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        with open(history_file, 'r') as log_file:
            lines = [line for line in log_file.readlines() if line.strip()]
            if len(lines) > 1:
                last_line = lines[-1]
                last_log = [float(i) for i in last_line.strip().split(",")]
                session_data["final_train_loss"] = last_log[1]
                session_data["final_val_loss"] = last_log[3]
            else:
                session_data["final_train_loss"] = -1
                session_data["final_val_loss"] = -1
    else:
        session_data["final_train_loss"] = -1
        session_data["final_val_loss"] = -1

    json_model = model.to_json()
    model_dict = json.loads(json_model)
    layers_list = model_dict['config']['layers']
    
    layers_data = [layer['class_name'] for layer in layers_list]
    session_data["architecture"] = '-'.join(layers_data)

    for set_name in ["training", "validation"]:
        if flights_summary.get(set_name) and len(flights_summary[set_name]) > 0:
            errors = np.array(flights_summary[set_name])[:, 2] 
            session_data[f"{set_name}_mean_vel_error"] = np.mean(errors)
            
            summary_file = os.path.join(trial_tree["trial_root_folder"], f"{set_name}_summary.csv")
            header = "Flight_ID,Duration_min,Max_Velocity_Error_m_s"
            np.savetxt(summary_file, flights_summary[set_name], delimiter=",", header=header, fmt='%s')

    summary_csv_path = os.path.join(project_root, 'summary.csv')
    write_header = not os.path.exists(summary_csv_path)
    
    fieldnames = sorted(session_data.keys())
    with open(summary_csv_path, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(session_data)