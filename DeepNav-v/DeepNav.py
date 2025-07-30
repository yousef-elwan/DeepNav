"""
DeepNav: Script to train and evaluate a velocity prediction model using IMU data.
This script uses a modern structure with argparse and TensorFlow's best practices.
"""

import os
import argparse
import tensorflow as tf

# Suppress extensive TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define the project's root directory once at the top
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Import custom modules
import utils
import postprocessing
from preprocessing.create_dataset import create_dataset


# ### تم التعديل: إضافة دوال الخسارة المخصصة هنا
def weighted_mae(y_true, y_pred):
    """
    Calculates Mean Absolute Error weighted by the magnitude of the true values.
    This gives more importance to errors on small true values.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = 1e-6 
    weights = 1.0 / (tf.math.abs(y_true) + epsilon)
    return tf.reduce_mean(weights * tf.math.abs(y_true - y_pred))

def wmae_mse_loss(y_true, y_pred):
    """
    The final combined loss function of MSE and Weighted MAE.
    An alpha of 0.5 gives equal importance to both losses.
    """
    alpha = 0.5
    
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    wmae = weighted_mae(y_true, y_pred)
    
    return alpha * mse + (1.0 - alpha) * wmae
# ### نهاية قسم دوال الخسارة


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
    """Main function to run the training and evaluation pipeline."""
    setup_gpu()

    # --- 1. Configuration ---
    column_names = {
        "features": ["a_x", "a_y", "a_z", "T", "q0", "q1", "q2", "q3"],
        "features_diff": ["h","Vn", "Ve", "Vd"],
        "labels": ["Pn", "Pe", "Pd"]
    }
    
    n_features = len(column_names["features"]) + len(column_names.get("features_diff", []))
    n_labels = len(column_names["labels"])

    session_mode = ["Fresh", "Resume", "Evaluate", "Override"]
    session_data = {
        "trial_number": args.trial,
        "session_mode": session_mode[args.mode],
        "gpu_name": f"/GPU:{args.gpu}" if args.gpu is not None else None,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "window_size": args.window,
        "epochs": args.epochs,
        "n_features": n_features,
        "n_labels": n_labels
    }

    print("\n--- Starting Session with Config ---")
    for key, value in session_data.items():
        print(f"{key:<20}: {value}")
    print("------------------------------------")
    
    trial_tree = utils.create_trial_tree(PROJECT_ROOT, session_data["trial_number"], session_data["session_mode"])

    # --- 2. Load or Create Dataset ---
    if args.new_dataset:
        session_data["dataset_name"] = None
    else:
        session_data["dataset_name"] = args.dataset_name

    train_ds, val_ds, train_flights_dict, val_flights_dict = create_dataset(
        PROJECT_ROOT, session_data, column_names
    )

    # --- 3. Prepare tf.data pipeline ---
    train_dataset = train_ds.shuffle(10000).batch(session_data["batch_size"]).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_ds.batch(session_data["batch_size"]).prefetch(tf.data.AUTOTUNE)

    # --- 4. Define Model Architecture ---
    print("\nDefining model (with Normalization, BiLSTM, Dropout)...")
    normalizer = tf.keras.layers.Normalization(axis=-1)
    if any(train_dataset):
        normalizer.adapt(train_dataset.map(lambda x, y: x))

    inputs = tf.keras.layers.Input(shape=(session_data["window_size"], session_data["n_features"]))
    x = normalizer(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=False)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_labels, activation='linear', name='velocity_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepNav_VelocityModel')
    model.summary()

    # --- 5. Compile with Gradient Clipping ---
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=session_data["learning_rate"],
        clipnorm=1.0
    )
    
    # ### تم التعديل: استخدام دالة الخسارة المخصصة
    model.compile(
        optimizer=optimizer,
        loss=wmae_mse_loss,
        metrics=['mae', 'mse'] # مراقبة المكونات الفردية للخسارة
    )

    # --- 6. Setup Callbacks ---
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(trial_tree["weights_folder"], 'ep.{epoch:03d}-val_loss_{val_loss:.4f}.weights.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(trial_tree["history_csv_file"], append=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=25, verbose=1, restore_best_weights=True
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # --- 7. Training or Evaluation ---
    if session_data["session_mode"] in ["Fresh", "Override"]:
        print(f"\nStarting fresh training for trial #{session_data['trial_number']}...")
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=session_data["epochs"],
            callbacks=callbacks
        )
    elif session_data["session_mode"] == "Resume":
        print(f"\nResuming training for trial #{session_data['trial_number']}...")
        latest_ckpt, __ = utils.retrieve_latest_weights(trial_tree["weights_folder"])
        if latest_ckpt:
            model.load_weights(latest_ckpt)
            print(f"Resumed from checkpoint: {latest_ckpt}")
        else:
            print("No checkpoint found to resume from. Starting fresh.")
        
        __, initial_epoch = utils.retrieve_latest_weights(trial_tree["weights_folder"])
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=session_data["epochs"],
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )
    else: # Evaluate mode
        latest_ckpt, __ = utils.retrieve_latest_weights(trial_tree["weights_folder"])
        if latest_ckpt:
            model.load_weights(latest_ckpt)
            print(f"\nModel loaded for evaluation from: {latest_ckpt}")
        else:
            print("ERROR: No model found to evaluate.")
            return

    # --- 8. Post-processing and Evaluation ---
    print("\nEvaluating model on all flights...")
    signal_meta = {
        "names": ["North Velocity (m/s)", "East Velocity (m/s)", "Down Velocity (m/s)"],
        "dt": 0.2
    }
    flights_summary = postprocessing.evaluate_all_flights(
        model=model,
        train_flights_dict=train_flights_dict,
        val_flights_dict=val_flights_dict,
        trial_folder=trial_tree["trial_root_folder"],
        signal_meta=signal_meta,
        n_extreme_flights=10
    )

    # --- 9. Summarize and Save ---
    print("\nSummarizing session and saving final model...")
    postprocessing.summarize_session(PROJECT_ROOT, trial_tree, model, session_data, flights_summary)
    
    keras_model_path = os.path.join(trial_tree["trial_root_folder"], "final_model.keras")
    model.save(keras_model_path)
    print(f"\nFinal Keras model saved to: {keras_model_path}")

    print("\nScript execution finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepNav: Train or evaluate a velocity prediction model.")
    parser.add_argument('--trial', type=int, default=1, help='Identifier for the current training trial.')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Session Mode: 0=Fresh, 1=Resume, 2=Evaluate, 3=Override.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--window', type=int, default=60, help='Size of the input window for sequence data.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=3500, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for the optimizer.')
    parser.add_argument('--new_dataset', action='store_true', help='Flag to force creation of a new dataset.')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset folder to use if not creating a new one.')
    
    args = parser.parse_args()
    main(args)