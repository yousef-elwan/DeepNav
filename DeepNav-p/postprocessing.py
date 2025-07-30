import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def flight_pdf_plots(file_name, ground_truth_abs_p, predicted_abs_p, signal_meta):
    time_steps_abs = np.arange(ground_truth_abs_p.shape[0]) * signal_meta.get("dt", 0.01)
    with PdfPages(file_name) as pdf:
        plt.figure(figsize=(12, 10))
        for i, name in enumerate(signal_meta["names"]):
            plt.subplot(3, 1, i + 1)
            plt.plot(time_steps_abs, ground_truth_abs_p[:, i], label="Ground Truth")
            plt.plot(time_steps_abs, predicted_abs_p[:, i], linestyle='--', label="Predicted")
            mae = np.mean(np.abs(ground_truth_abs_p[:, i] - predicted_abs_p[:, i]))
            plt.grid(True); plt.legend(); plt.xlabel("Time (s)"); plt.ylabel(f"{name} (m)")
            plt.title(f"Position Track: {name} | MAE: {mae:.2f} m")
        plt.suptitle("Position Tracks: Ground Truth vs. Predicted", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(); plt.close()

        plt.figure(figsize=(8, 8))
        plt.plot(ground_truth_abs_p[:, 1], ground_truth_abs_p[:, 0], label="Ground Truth")
        plt.plot(predicted_abs_p[:, 1], predicted_abs_p[:, 0], linestyle='--', label="Predicted")
        plt.grid(True); plt.legend(); plt.xlabel("East (m)"); plt.ylabel("North (m)")
        plt.title("2D Flight Path (Top-Down View: Pe vs Pn)")
        plt.axis('equal')
        pdf.savefig(); plt.close()


def evaluate_all_flights(model, train_flights_dict, val_flights_dict, trial_folder, signal_meta):
    # --- KEY CHANGE: Define the same scaling factor ---
    LABEL_SCALING_FACTOR = 10.0
    # ----------------------------------------------
    
    plots_dir = os.path.join(trial_folder, "plots", "other")
    os.makedirs(plots_dir, exist_ok=True)
    
    for set_name, flights_dict in zip(["training", "validation"], [train_flights_dict, val_flights_dict]):
        print(f"\nProcessing {set_name} set...")
        if not flights_dict: continue
            
        for i, (flight_name, (features, ground_truth_abs_seq)) in enumerate(sorted(flights_dict.items())):
            print(f"  flight {i+1}/{len(flights_dict)}: {flight_name}", end="")
            
            if features.shape[0] == 0 or ground_truth_abs_seq.shape[0] == 0:
                print(" | Skipping due to empty data.")
                continue

            # Model predicts the SCALED deltas
            predicted_delta_seq_scaled = model.predict(features, verbose=0)
            
            # --- KEY CHANGE: Reverse the scaling on the predictions ---
            predicted_delta_seq = predicted_delta_seq_scaled / LABEL_SCALING_FACTOR
            # --------------------------------------------------------
            
            start_position = ground_truth_abs_seq[0]
            cumulative_deltas = np.cumsum(predicted_delta_seq, axis=0)
            
            predicted_abs_seq = np.vstack([
                start_position,
                start_position + cumulative_deltas[:-1]
            ])

            min_len = min(len(ground_truth_abs_seq), len(predicted_abs_seq))
            ground_truth_abs_seq = ground_truth_abs_seq[:min_len]
            predicted_abs_seq = predicted_abs_seq[:min_len]

            mean_error = np.mean(np.abs(ground_truth_abs_seq - predicted_abs_seq), axis=0)
            print(f" | MAE (Pn,Pe,Pd): {mean_error[0]:.2f}m, {mean_error[1]:.2f}m, {mean_error[2]:.2f}m")
            
            pdf_name = f"{flight_name}_POS_EVAL.pdf"
            pdf_path = os.path.join(plots_dir, pdf_name)
            flight_pdf_plots(pdf_path, ground_truth_abs_seq, predicted_abs_seq, signal_meta)