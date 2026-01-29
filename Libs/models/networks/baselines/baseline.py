import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Dict, Any, Tuple, Optional


class BaselineModel:
    """
    Baseline model using gradient boosting trees (XGBoost/LightGBM) for fatigue life prediction.
    """
    
    def __init__(self, model_type='lightgbm', random_state=42):
        """
        Initialize the baseline model.
        
        Args:
            model_type: Type of model to use ('xgboost' or 'lightgbm').
            random_state: Random seed for reproducibility.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares features and labels from the training dataframe.
        
        Args:
            df: DataFrame with features and labels.
        
        Returns:
            Tuple of (features_df, labels_series).
        """
        # Exclude non-feature columns
        exclude_cols = ['rul', 'fatigue_life', 'cycle_index']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features = df[feature_cols].copy()
        labels = df['rul'].copy()
        
        # Remove rows with NaN labels
        valid_mask = ~labels.isna()
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        return features, labels
    
    def train(self, df: pd.DataFrame, test_size=0.2, validation_size=0.2, **kwargs):
        """
        Trains the baseline model with train-validation-test split.
        
        Args:
            df: Training dataframe with features and labels.
            test_size: Fraction of data to use for testing.
            validation_size: Fraction of data to use for validation (from training set).
            **kwargs: Additional hyperparameters for the model.
        """
        features, labels = self.prepare_features(df)
        self.feature_names = features.columns.tolist()
        
        # Split data into train and test first
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=self.random_state
        )
        
        # Split train_val into train and validation
        val_size_from_train = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_from_train, random_state=self.random_state
        )
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'max_depth', 'learning_rate']}
            )
        else:  # lightgbm
            self.model = lgb.LGBMRegressor(
                random_state=self.random_state,
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                verbose=-1,
                **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'max_depth', 'learning_rate']}
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on all three sets
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\n{self.model_type.upper()} Baseline Model Performance:")
        print(f"Train RMSE: {train_rmse:.2f}, Val RMSE: {val_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        print(f"Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}, Test MAE: {test_mae:.2f}")
        print(f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}")
        
        return {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': test_pred
        }
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model.
        
        Args:
            features: DataFrame with features.
        
        Returns:
            Array of predictions.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(features)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Gets feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        else:  # lightgbm
            importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Saves the trained model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Loads a trained model from disk.
        
        Args:
            filepath: Path to load the model from.
        """
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.feature_names = loaded['feature_names']
        self.model_type = loaded['model_type']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


if __name__ == '__main__':
    from Libs.data.dataloader import load_arrows_data
    from Libs.data.physics_processor import PhysicsModel
    from Libs.data.label_generator import FatigueLabelGenerator
    
    print("Loading ARROWS data...")
    data = load_arrows_data('Input/raw/data.mat')
    
    if data:
        print("Computing elastic modulus...")
        physics = PhysicsModel()
        modulus_result = physics.compute_elastic_modulus(data['sensors'], data['time'])
        
        print("Generating training dataset...")
        label_gen = FatigueLabelGenerator()
        training_df = label_gen.prepare_training_data(
            modulus_result, 
            data['sensors'], 
            data['time']
        )
        
        print(f"\nTraining dataset shape: {training_df.shape}")
        
        # Train LightGBM model
        print("\nTraining LightGBM baseline model...")
        baseline_lgb = BaselineModel(model_type='lightgbm')
        results_lgb = baseline_lgb.train(training_df, test_size=0.3)
        
        # Show feature importance
        print("\nFeature Importance:")
        print(baseline_lgb.get_feature_importance())

