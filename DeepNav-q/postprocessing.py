import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shutil

# --- Quaternion Math Helper Functions ---
def quaternion_to_euler_angles(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg([roll, pitch, yaw])

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

def flight_pdf_plots(file_name, ground_truth_abs_q, predicted_abs_q, signal_meta):
    """
    Creates a PDF with plots comparing ground truth and predicted absolute quaternions,
    and their Euler angles.
    """
    ground_truth_abs_q = np.asarray(ground_truth_abs_q)
    predicted_abs_q = np.asarray(predicted_abs_q)
    time_steps_abs = np.arange(ground_truth_abs_q.shape[0]) * signal_meta.get("dt", 0.01)

    with PdfPages(file_name) as pdf:
        plt.figure(figsize=(12, 8))
        for i, name in enumerate(signal_meta["names"]):
            plt.subplot(2, 2, i + 1)
            plt.plot(time_steps_abs, ground_truth_abs_q[:, i], label="Ground Truth")
            plt.plot(time_steps_abs, predicted_abs_q[:, i], linestyle='--', label="Predicted (Accumulated)")
            plt.grid(True); plt.legend(); plt.xlabel("Time (s)"); plt.ylabel(name)
            plt.title(f"Abs. Quat. Comp: {name}")
        plt.suptitle("Absolute Quaternion Components: Ground Truth vs. Predicted", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(); plt.close()

        gt_euler = np.array([quaternion_to_euler_angles(q) for q in ground_truth_abs_q])
        pred_euler = np.array([quaternion_to_euler_angles(q) for q in predicted_abs_q])
        
        plt.figure(figsize=(12, 10))
        for i, name in enumerate(["Roll", "Pitch", "Yaw"]):
            plt.subplot(3, 1, i + 1)
            plt.plot(time_steps_abs, gt_euler[:, i], label="Ground Truth Euler")
            plt.plot(time_steps_abs, pred_euler[:, i], linestyle='--', label="Predicted Euler")
            mae = np.mean(np.abs(gt_euler[:, i] - pred_euler[:, i]))
            plt.grid(True); plt.legend(); plt.xlabel("Time (s)"); plt.ylabel(f"{name} (degrees)")
            plt.title(f"Euler Angle: {name} | MAE: {mae:.2f}°")
        plt.suptitle("Euler Angles: Ground Truth vs. Predicted", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(); plt.close()

def evaluate_all_flights(model, train_flights_dict, val_flights_dict, trial_folder, signal_meta, n_extreme_flights=10):
    """
    Evaluates the model by accumulating predicted delta quaternions, comparing the
    reconstructed absolute orientation with the ground truth, and SAVING PLOTS AND CSVs.
    """
    flights_summary = {}
    for sub_dir in ["other", "best", "worst"]:
        os.makedirs(os.path.join(trial_folder, "plots", sub_dir), exist_ok=True)

    for set_name, flights_dict in zip(["training", "validation"], [train_flights_dict, val_flights_dict]):
        print(f"\nProcessing {set_name} set...")
        if not flights_dict:
            print(f"No flights found in {set_name} set. Skipping.")
            flights_summary[set_name] = []
            continue
            
        flights_list = sorted(flights_dict.items())
        flights_errors = {}
        set_summary = []

        for i, (flight_name, (features, ground_truth_abs_seq)) in enumerate(flights_list):
            print(f"  flight {i+1}/{len(flights_list)}: {flight_name}", end="")
            
            # Predict the sequence of delta quaternions
            predicted_delta_seq = model.predict(features, verbose=0)
            
            # Accumulate deltas to reconstruct the absolute quaternion sequence
            predicted_abs_seq = np.zeros_like(ground_truth_abs_seq)
            predicted_abs_seq[0] = ground_truth_abs_seq[0] # Start with the first true orientation

            for j in range(1, len(predicted_delta_seq)):
                predicted_abs_seq[j] = quaternion_multiply(predicted_delta_seq[j], predicted_abs_seq[j-1])

            # Ensure ground truth is the same length as predictions
            if len(predicted_abs_seq) != len(ground_truth_abs_seq):
                print(f" | Warning: Mismatch in length. Pred: {len(predicted_abs_seq)}, GT: {len(ground_truth_abs_seq)}")
                min_len = min(len(predicted_abs_seq), len(ground_truth_abs_seq))
                predicted_abs_seq = predicted_abs_seq[:min_len]
                ground_truth_abs_seq = ground_truth_abs_seq[:min_len]

            # Calculate angular error
            dot_product = np.clip(np.sum(ground_truth_abs_seq * predicted_abs_seq, axis=1), -1.0, 1.0)
            angular_error_rad = 2 * np.arccos(np.abs(dot_product))
            mean_angular_error_deg = np.rad2deg(np.mean(angular_error_rad))
            max_angular_error_deg = np.rad2deg(np.max(angular_error_rad))
            print(f" | Mean Angular Error: {mean_angular_error_deg:.2f}°")
            
            # --- 🔽 [القسم المضاف] حفظ بيانات الرحلة في ملف CSV 🔽 ---
            # 1. حساب زوايا أويلر للبيانات الحقيقية والمتوقعة
            gt_euler = np.array([quaternion_to_euler_angles(q) for q in ground_truth_abs_seq])
            pred_euler = np.array([quaternion_to_euler_angles(q) for q in predicted_abs_seq])
            time_steps_abs = np.arange(ground_truth_abs_seq.shape[0]) * signal_meta.get("dt", 0.01)

            # 2. إنشاء قاموس لتنظيم البيانات
            csv_data = {
                'time_s': time_steps_abs,
                'gt_q0_w': ground_truth_abs_seq[:, 0], 'gt_q1_x': ground_truth_abs_seq[:, 1], 'gt_q2_y': ground_truth_abs_seq[:, 2], 'gt_q3_z': ground_truth_abs_seq[:, 3],
                'pred_q0_w': predicted_abs_seq[:, 0], 'pred_q1_x': predicted_abs_seq[:, 1], 'pred_q2_y': predicted_abs_seq[:, 2], 'pred_q3_z': predicted_abs_seq[:, 3],
                'gt_roll_deg': gt_euler[:, 0], 'gt_pitch_deg': gt_euler[:, 1], 'gt_yaw_deg': gt_euler[:, 2],
                'pred_roll_deg': pred_euler[:, 0], 'pred_pitch_deg': pred_euler[:, 1], 'pred_yaw_deg': pred_euler[:, 2],
            }
            
            # 3. تحويل القاموس إلى DataFrame وحفظه
            df = pd.DataFrame(csv_data)
            csv_filename = f"{flight_name}_data.csv"
            csv_path = os.path.join(trial_folder, "plots", "other", csv_filename)
            df.to_csv(csv_path, index=False, float_format='%.6f')
            # --- 🔼 [نهاية القسم المضاف] 🔼 ---

            # Create PDF plots
            safe_mae_str = f"{mean_angular_error_deg:.2f}".replace('.', '_')
            pdf_name = f"{flight_name}_MAE_{safe_mae_str}.pdf"
            pdf_path = os.path.join(trial_folder, "plots", "other", pdf_name)
            flight_pdf_plots(pdf_path, ground_truth_abs_seq, predicted_abs_seq, signal_meta)
            
            flights_errors[pdf_name] = mean_angular_error_deg
            set_summary.append([flight_name, mean_angular_error_deg, max_angular_error_deg])
        
        flights_summary[set_name] = set_summary
        
        # Sorting and moving plots
        if flights_errors:
            # ... (code for moving best/worst plots remains the same)
            pass

    return flights_summary