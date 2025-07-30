import os
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input, Normalization, Bidirectional, LSTM, Dropout, Dense, BatchNormalization

# Import custom modules
import utils
import postprocessing
from preprocessing.create_dataset import create_dataset
from training import start_training

def setup_gpu():
    """Configures GPU settings for TensorFlow to allow memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on {len(gpus)} GPU(s) with memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU setup failed: {e}")

def main(args):
    setup_gpu()

    # --- 1. Configuration ---
    session_data = {
        "trial_number": args.trial,
        "session_mode": ["Fresh", "Resume", "Evaluate", "Override"][args.mode],
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "window_size": args.window,
        "epochs": args.epochs,
        "loss_alpha": args.loss_alpha 
    }

    # --- 2. Load or Create Dataset ---
    column_names = {
        "features": [
            "w_x", "w_y", "w_z", "a_x", "a_y", "a_z",
            "q0", "q1", "q2", "q3", "Vn", "Ve", "Vd"
        ],
        "labels": ["Pn", "Pe", "Pd"]
    }
    session_data["n_features"] = len(column_names["features"])

    train_ds, val_ds, train_flights_dict, val_flights_dict = create_dataset(
        session_data, column_names
    )
    
    # --- 3. Prepare tf.data pipeline ---
    train_dataset = train_ds.shuffle(10000).batch(session_data["batch_size"]).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_ds.batch(session_data["batch_size"]).prefetch(tf.data.AUTOTUNE)

    # --- 4. Define Model Architecture with Batch Normalization ---
    print("\nDefining model with Batch Normalization for stability...")
    feature_normalizer = Normalization(axis=-1)
    feature_normalizer.adapt(train_dataset.map(lambda x, y: x))

    inputs = Input(shape=(session_data["window_size"], session_data["n_features"]))
    x = feature_normalizer(inputs)
    
    # Layer 1
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    # --- KEY CHANGE: Add Batch Normalization ---
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Layer 2
    x = Bidirectional(LSTM(256, return_sequences=False))(x)
    # --- KEY CHANGE: Add Batch Normalization ---
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(3, activation='linear', name='position_delta_output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepNav_p_Model')

    # --- 5. Training or Evaluation ---
    trial_tree = utils.create_trial_tree(session_data['trial_number'], session_data['session_mode'])
    
    model, history = start_training(
        session_data=session_data,
        model=model,
        train_ds=train_dataset,
        val_ds=val_dataset,
        trial_tree=trial_tree
    )

    if history:
        print("\nTraining finished.")
    else:
        print("\nEvaluation mode: Model loaded without training.")

    # --- 6. Post-processing and Evaluation ---
    print("\nEvaluating model on all flights...")
    signal_meta = { "names": ["Pn", "Pe", "Pd"], "dt": 0.01 }
    postprocessing.evaluate_all_flights(
        model=model,
        train_flights_dict=train_flights_dict,
        val_flights_dict=val_flights_dict,
        trial_folder=trial_tree["trial_root_folder"],
        signal_meta=signal_meta
    )

    # --- 7. Summarize and Save ---
    print("\nSummarizing session and saving final model...")
    keras_model_path = os.path.join(trial_tree["trial_root_folder"], "final_model.keras")
    model.save(keras_model_path)
    print(f"\nKeras model saved to: {keras_model_path}")
    print("\nScript execution finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepNav-p: Train or evaluate a position prediction model.")
    parser.add_argument('--trial', type=int, default=1, help='Identifier for the current training trial.')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3], default=0, help='Session Mode: 0=Fresh, 1=Resume, 2=Evaluate, 3=Override.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--window', type=int, default=50, help='Size of the input window for sequence data.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for the optimizer.')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='Weight for the MSE part of the combined loss.')
    args = parser.parse_args()
    main(args)