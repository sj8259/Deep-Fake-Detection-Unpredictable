# 🎭 Deepfake Image Classification using InceptionResNetV2


---
## 🏗 Model Architecture
The model is built on **InceptionResNetV2** (pre-trained on ImageNet) with the following layers added:

1️⃣ **GlobalAveragePooling2D**  
2️⃣ **Dense(1024, activation='relu')**  
3️⃣ **Dense(1, activation='sigmoid')**

---
## 🔧 Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install tensorflow numpy opencv-python scikit-learn
```

---
## 🏋️ Training Pipeline
### 1️⃣ Mount Google Drive & Extract Dataset
```python
from google.colab import drive
drive.mount('/content/drive')
```
### 2️⃣ Load and Preprocess Images
- Resize images to **224x224**
- Normalize pixel values between **0 and 1**
### 3️⃣ Split Dataset
- **80% Training**
- **20% Testing**
### 4️⃣ Apply **Data Augmentation** using `ImageDataGenerator`
### 5️⃣ Train Model for **20 Epochs**
- Optimizer: **Adam**
- Loss Function: **Binary Cross-Entropy**
### 6️⃣ Evaluate Model Performance
- Compute final **accuracy score**

---
## 📈 Model Training
```python
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20
)
```

---
## 📊 Evaluation
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {test_acc:.4f}")
```

---
## 🔍 Inference
To classify new test images:
```python
test_predictions = model.predict(test_images)
test_predictions = (test_predictions > 0.5).astype(int)
```
Results are saved in **Unpredictable.json**:
```json
[
    {"index": 1, "prediction": "real"},
    {"index": 2, "prediction": "fake"}
]
```

