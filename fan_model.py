"""
Feature 1: Temperature Based Fan Speed Predictor
Uses Decision Tree Classifier from Scikit-learn
Dataset: fan_speed_dataset.csv (111 rows, 2 columns)
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

class FanSpeedPredictor:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.speed_labels = {
            1: {"range": "0°C - 15°C", "label": "Very Low"},
            2: {"range": "16°C - 20°C", "label": "Low"},
            3: {"range": "21°C - 25°C", "label": "Medium"},
            4: {"range": "26°C - 30°C", "label": "High"},
            5: {"range": "31°C+", "label": "Very High"}
        }

    def train(self, dataset_path=None):
        """Train the Decision Tree model on fan speed dataset."""
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(__file__), 'fan_speed_dataset.csv')

        try:
            df = pd.read_csv(dataset_path)
            X = df[['temperature']].values
            y = df['fan_speed'].values

            self.model = DecisionTreeClassifier(random_state=42)
            self.model.fit(X, y)
            self.is_trained = True
            print(f"[Fan Model] Trained on {len(df)} samples. Ready!")
            return True
        except Exception as e:
            print(f"[Fan Model] Training error: {e}")
            return False

    def predict(self, temperature):
        """Predict fan speed for a given temperature."""
        if not self.is_trained:
            return None

        try:
            temp = float(temperature)
            # Clamp temperature to reasonable range
            temp = max(0, min(60, temp))
            prediction = self.model.predict([[temp]])[0]
            speed = int(prediction)

            return {
                "speed": speed,
                "temperature": temp,
                "range": self.speed_labels[speed]["range"],
                "label": self.speed_labels[speed]["label"]
            }
        except Exception as e:
            print(f"[Fan Model] Prediction error: {e}")
            return None


# Singleton instance
fan_predictor = FanSpeedPredictor()
