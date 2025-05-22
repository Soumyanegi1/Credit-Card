

import numpy as np
import pandas as pd
import os
import cv2
import pickle
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, LeakyReLU
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.linear_model import Ridge

# Load dataset
df = pd.read_csv("enhanced_credit_score_data.csv")
df_numeric = df.drop(columns=["CUST_ID", "CAT_GAMBLING"])
X = df_numeric.drop(columns=["CREDIT_SCORE", "DEFAULT"])
Y = df_numeric["CREDIT_SCORE"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimize PCA
pca = PCA(n_components=0.999, svd_solver='auto')
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Heatmap Directory
heatmap_dir = "heatmaps"
os.makedirs(heatmap_dir, exist_ok=True)

def reshape_to_grid(record, target_size=(32, 32)):
    padded = np.pad(record, (0, target_size[0] * target_size[1] - len(record)), mode='constant')
    return np.reshape(padded, target_size)

def save_heatmap(index):
    record = X_test_pca[index]
    reshaped_record = reshape_to_grid(record).astype(np.float32)
    heatmap = cv2.normalize(reshaped_record, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(heatmap_dir, f"record_{index}.png"), heatmap)

if __name__ == "__main__":
    with Pool(processes=cpu_count()) as pool:
        pool.map(save_heatmap, range(len(X_test_pca)))

    def load_and_preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(32, 32), color_mode="grayscale")
        return image.img_to_array(img) / 255.0  
    
    heatmap_features = np.array([load_and_preprocess_image(os.path.join(heatmap_dir, f"record_{i}.png")) for i in range(len(X_test_pca))])

    # Enhanced CNN Model
    def build_cnn_model():
        model = Sequential([
            Conv2D(128, (3, 3), input_shape=(32, 32, 1), kernel_regularizer='l2'),
            LeakyReLU(alpha=0.1), BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.4),
            Conv2D(256, (3, 3), kernel_regularizer='l2'), LeakyReLU(alpha=0.1), BatchNormalization(), MaxPooling2D((2, 2)), Dropout(0.4),
            Conv2D(512, (3, 3), kernel_regularizer='l2'), LeakyReLU(alpha=0.1), BatchNormalization(), Flatten(), Dropout(0.4),
            Dense(1024), LeakyReLU(alpha=0.1), Dropout(0.4), Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Train CNN
    cnn_model = build_cnn_model()
    cnn_model.fit(heatmap_features, Y_test, epochs=100, batch_size=64, verbose=1)
    cnn_features = cnn_model.predict(heatmap_features)

    # Bagging Model
    bagging_model = BaggingRegressor(n_estimators=150, max_samples=0.9, max_features=0.9, random_state=42)
    bagging_model.fit(cnn_features, Y_test)
    y_pred_bagging = bagging_model.predict(cnn_features)

    # XGBoost Model
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=12, subsample=0.9, colsample_bytree=0.9, random_state=42)
    xgb_model.fit(cnn_features, Y_test)
    y_pred_xgb = xgb_model.predict(cnn_features)

    # Meta-Model for Ensemble
    meta_model = Ridge(alpha=1.0)
    meta_features = np.column_stack((y_pred_bagging, y_pred_xgb))
    meta_model.fit(meta_features, Y_test)
    final_predictions = meta_model.predict(meta_features)

    # Evaluation
    mae_final = mean_absolute_error(Y_test, final_predictions)
    r2_final = r2_score(Y_test, final_predictions)
    
    print(f"MAE: {mae_final}")
    print(f"R2 Score: {r2_final}")

    # Save models
    with open("trained_models.pkl", "wb") as f:
        pickle.dump({"cnn_model": cnn_model, "bagging_model": bagging_model, "xgb_model": xgb_model, "meta_model": meta_model}, f)

    print("Models saved as trained_models.pkl")


# Load the trained model and make predictions on new input
def load_and_test_model():
    with open("trained_models.pkl", "rb") as f:
        models = pickle.load(f)

    cnn_model = models["cnn_model"]
    bagging_model = models["bagging_model"]
    xgb_model = models["xgb_model"]
    meta_model = models["meta_model"]

    # Example input
    new_input = np.random.rand(1, 32, 32, 1)  # Replace this with an actual heatmap image
    cnn_feature = cnn_model.predict(new_input)
    
    y_pred_bagging = bagging_model.predict(cnn_feature)
    y_pred_xgb = xgb_model.predict(cnn_feature)
    
    meta_input = np.column_stack((y_pred_bagging, y_pred_xgb))
    final_prediction = meta_model.predict(meta_input)
    
    print(f"Predicted Credit Score: {final_prediction[0]}")

# Call this function after training to test the model with new input
# Uncomment the line below to test
# load_and_test_model()
