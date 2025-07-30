"""
This script is the training backend, called by DeepNav.py.
"""

import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils import retrieve_latest_weights

def quaternion_angular_distance(y_true, y_pred):
    """
    تحسب المسافة الزاويّة (geodesic distance) بين اثنين من الكواترنيون.
    الخسارة هي زاوية الدوران اللازمة للانتقال من y_pred إلى y_true.

    Args:
        y_true: الكواترنيون الحقيقي (ground truth).
        y_pred: الكواترنيون الذي تم توقعه بواسطة النموذج.

    Returns:
        قيمة الخسارة (الخطأ الزاوي بالراديان).
    """
    # التأكد من أن الكواترنيونات هي متجهات وحدة (unit quaternions)
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    
    # حساب حاصل الضرب النقطي (dot product)
    # نستخدم القيمة المطلقة للتعامل مع حقيقة أن q و -q يمثلان نفس الدوران
    dot_product = tf.abs(tf.reduce_sum(y_true * y_pred, axis=-1))
    
    # حاصل الضرب النقطي يساوي cos(theta/2)، حيث theta هي الزاوية بين الدورانين.
    # لتجنب الأخطاء الرقمية عند استخدام acos، نتأكد من أن القيمة بين -1 و 1
    clipped_dot_product = tf.clip_by_value(dot_product, -1.0, 1.0)
    
    # حساب نصف الزاوية
    half_angle = tf.acos(clipped_dot_product)
    
    # الزاوية الكاملة (الخطأ بالراديان) هي ضعف نصف الزاوية
    angular_error = 2.0 * half_angle
    
    return angular_error

# <-- تعديل: إضافة `strategy` كمعامل للدالة
def start_training(session_data, model_architecture, train_ds, val_ds, trial_tree):
    """
    Compiles and fits the model. Returns the trained model and history.
    """
    weights_folder = trial_tree["weights_folder"]
    history_csv_file = trial_tree["history_csv_file"]
    initial_epoch = 0

    # <-- تعديل: حذف إنشاء الاستراتيجية من هنا، لأننا نستلمها كمعامل
    # strategy = tf.distribute.OneDeviceStrategy(...)
        # إنشاء النموذج
    model = tf.keras.models.Sequential(model_architecture)
        
        # تعريف المُحسِّن
    optimizer = Adam(learning_rate=session_data["learning_rate"])
        
        # تجميع النموذج
    model.compile(loss=quaternion_angular_distance, optimizer=optimizer, metrics=["mae"])
    model.build(input_shape=(None, session_data["window_size"], session_data["n_features"]))

        # <-- تعديل: نقل منطق تحميل الأوزان إلى داخل نطاق الاستراتيجية
    if session_data["session_mode"] in ["Resume", "Evaluate"]:
        print("\nSearching for latest weights...")
        initial_epoch, latest_weights = retrieve_latest_weights(weights_folder)
        if latest_weights:
            model.load_weights(latest_weights)
            print(f"Successfully loaded weights from: {latest_weights}")
            print(f"Starting from epoch: {initial_epoch}")
        elif session_data["session_mode"] == "Evaluate":
            raise FileNotFoundError("Cannot evaluate. No saved weights found in " + weights_folder)

    # طباعة ملخص النموذج تتم خارج النطاق
    model.summary()

    # تعريف الـ Callbacks
    callbacks = [
        tf.keras.callbacks.CSVLogger(history_csv_file, append=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_folder, 'ep.{epoch:04d}.weights.h5'),
            save_weights_only=True,
            save_best_only=False,
            save_freq='epoch'
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    ]

    # بدء التدريب
    history = None
    start_train_time = time.time()
    if session_data["session_mode"] != "Evaluate":
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=session_data["epochs"],
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=2
        )
    
    session_data["training_time_hr"] = (time.time() - start_train_time) / 3600
    return model, history