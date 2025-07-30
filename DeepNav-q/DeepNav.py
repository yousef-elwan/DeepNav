import os
import argparse
import tensorflow as tf

# Suppress extensive TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import custom modules
import utils
import postprocessing
from preprocessing.create_dataset import create_dataset

class L2NormalizationLayer(tf.keras.layers.Layer):
    """
    Performs L2 normalization on the last axis of the input tensor.
    This custom layer saves and loads correctly without any issues.
    """
    def __init__(self, **kwargs):
        super(L2NormalizationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        # This allows the model to be saved and loaded correctly
        config = super(L2NormalizationLayer, self).get_config()
        return config
    

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
    session_mode = ["Fresh", "Resume", "Evaluate", "Override"]
    session_data = {
        "trial_number": args.trial,
        "session_mode": session_mode[args.mode],
        "gpu_name": f"/GPU:{args.gpu}" if args.gpu is not None else None,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "window_size": args.window,
        "epochs": args.epochs,
    }

    print("\n--- Starting Session with Config ---")
    for key, value in session_data.items():
        print(f"{key:<20}: {value}")
    print("------------------------------------")

    # --- 2. Load or Create Dataset ---
    if args.new_dataset:
        session_data["dataset_name"] = None
        column_names = {
            "features": ["w_x", "w_y", "w_z", "a_x", "a_y", "a_z", "m_x", "m_y", "m_z", "T"],
            "features_diff": ["h"],
            "labels": ["q0", "q1", "q2", "q3"]
        }
    else:
        session_data["dataset_name"] = "T001_logs548_F10L6_W50_03Dec2020_1542_FMUV5"
        column_names = {}

    train_ds, val_ds, train_flights_dict, val_flights_dict = create_dataset(
        session_data, column_names
    )
    session_data["n_features"] = args.n_features

    # --- 3. Prepare tf.data pipeline ---
    train_dataset = train_ds.shuffle(10000).batch(session_data["batch_size"]).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_ds.batch(session_data["batch_size"]).prefetch(tf.data.AUTOTUNE)

    # --- 4. Define Model Architecture ---
    print("\nDefining model (with Normalization, BiLSTM, Dropout) ...")
    # Normalization layer adapted on train data
    normalizer = tf.keras.layers.Normalization(axis=-1)
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
    x = tf.keras.layers.Dense(4, activation='linear')(x)
    outputs = L2NormalizationLayer(name='quaternion_output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DeepNavModel')

    # --- 5. Compile with Gradient Clipping ---
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=session_data["learning_rate"],
        clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # --- 6. Setup Callbacks ---
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --- 7. Training or Evaluation ---
    if session_data["session_mode"] in ["Fresh", "Resume"]:
        print(f"\nStarting training for trial #{session_data['trial_number']}...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=session_data["epochs"],
            callbacks=callbacks
        )
        print("\nTraining finished.")
    else:
        # Evaluate mode: load weights and skip training
        ckpt = utils.find_latest_checkpoint(
            utils.create_trial_tree(session_data['trial_number'], session_data['session_mode'])["trial_root_folder"]
        )
        model.load_weights(ckpt)
        print("\nModel loaded for evaluation.")

    # --- 8. Post-processing and Evaluation ---
    print("\nEvaluating model on all flights...")
    signal_meta = {
        "names": ["q0 (w)", "q1 (x)", "q2 (y)", "q3 (z)"],
        "dt": 0.004
    }
    flights_summary = postprocessing.evaluate_all_flights(
        model=model,
        train_flights_dict=train_flights_dict,
        val_flights_dict=val_flights_dict,
        trial_folder=utils.create_trial_tree(session_data["trial_number"], session_data["session_mode"])["trial_root_folder"],
        signal_meta=signal_meta,
        n_extreme_flights=10
    )

    # --- 9. Summarize and Save ---
    print("\nSummarizing session and saving final model...")
    trial_folder = utils.create_trial_tree(session_data["trial_number"], session_data["session_mode"])["trial_root_folder"]
    keras_model_path = os.path.join(trial_folder, "final_model.keras")
    model.save(keras_model_path)
    print(f"\nKeras model saved to: {keras_model_path}")

    print("\nScript execution finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepNav: Train or evaluate a quaternion prediction model.")
    parser.add_argument('--trial', type=int, default=1, help='Identifier for the current training trial.')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Session Mode: 0=Fresh, 1=Resume, 2=Evaluate, 3=Override.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--window', type=int, default=10, help='Size of the input window for sequence data.')
    parser.add_argument('--n_features', type=int, default=11, help='Number of input features in the dataset.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for the optimizer.')
    parser.add_argument('--new_dataset', action='store_true', default=True,
                        help='Flag to create a new dataset instead of using a cached one.')
    args = parser.parse_args()
    main(args)
