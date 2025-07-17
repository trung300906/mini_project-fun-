#!/usr/bin/env python3
"""
AI module with Deep Learning for phone number validation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, BatchNormalization, Attention
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re
import os
import joblib
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class PhoneSequenceDataset(Dataset):
    """PyTorch Dataset cho phone sequences"""
    
    def __init__(self, phone_numbers: List[str], labels: List[int], max_length: int = 15):
        self.phone_numbers = phone_numbers
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.phone_numbers)
    
    def __getitem__(self, idx):
        phone = self.phone_numbers[idx]
        label = self.labels[idx]
        
        # Chuyển đổi phone thành sequence
        clean_phone = re.sub(r'\D', '', phone)
        
        # Padding hoặc truncate
        if len(clean_phone) > self.max_length:
            clean_phone = clean_phone[:self.max_length]
        else:
            clean_phone = clean_phone.ljust(self.max_length, '0')
        
        # Chuyển thành tensor
        sequence = torch.tensor([int(d) for d in clean_phone], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return sequence, label

class PhoneTransformerModel(nn.Module):
    """Transformer model for phone validation"""
    
    def __init__(self, vocab_size=10, d_model=128, nhead=8, num_layers=4, max_length=15):
        super(PhoneTransformerModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(max_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )
    
    def _create_positional_encoding(self, max_length, d_model):
        """Tạo positional encoding"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1)].to(x.device)
        
        # Transformer
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=0)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x

class PhoneLSTMModel(nn.Module):
    """LSTM model for phone validation"""
    
    def __init__(self, vocab_size=10, embed_dim=64, hidden_dim=128, num_layers=2):
        super(PhoneLSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2 vì bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Sử dụng hidden state cuối
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Classification
        output = self.classifier(hidden)
        
        return output

class PhoneCNNModel(nn.Module):
    """CNN model for phone validation"""
    
    def __init__(self, vocab_size=10, embed_dim=64, num_filters=100):
        super(PhoneCNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multiple conv layers với filter sizes khác nhau
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, num_filters, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        # Conv layers
        conv1_out = torch.relu(self.conv1(x))
        conv2_out = torch.relu(self.conv2(x))
        conv3_out = torch.relu(self.conv3(x))
        
        # Global max pooling
        pool1 = self.pool(conv1_out).squeeze(-1)
        pool2 = self.pool(conv2_out).squeeze(-1)
        pool3 = self.pool(conv3_out).squeeze(-1)
        
        # Concatenate
        x = torch.cat([pool1, pool2, pool3], dim=1)
        x = self.dropout(x)
        
        # Classification
        output = self.classifier(x)
        
        return output

class PhoneAIValidator:
    """AI-powered phone validator"""
    
    def __init__(self, model_path: str = "phone_ai_models"):
        self.model_path = model_path
        self.models = {}
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        
        # TensorFlow models
        self.tf_models = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        # PyTorch models
        self.models['transformer'] = PhoneTransformerModel().to(self.device)
        self.models['lstm'] = PhoneLSTMModel().to(self.device)
        self.models['cnn'] = PhoneCNNModel().to(self.device)
        
        # TensorFlow models
        self._create_tf_models()
    
    def _create_tf_models(self):
        """Tạo TensorFlow models"""
        
        # Dense Neural Network
        self.tf_models['dense'] = Sequential([
            Dense(256, activation='relu', input_shape=(50,)),  # 50 features
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax')
        ])
        
        # LSTM for sequences
        self.tf_models['lstm_tf'] = Sequential([
            Embedding(10, 64, input_length=15),
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')
        ])
        
        # CNN for sequences
        self.tf_models['cnn_tf'] = Sequential([
            Embedding(10, 64, input_length=15),
            Conv1D(64, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')
        ])
        
        # Compile models
        for model in self.tf_models.values():
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def prepare_data(self, phone_numbers: List[str], labels: List[int]) -> Dict:
        """Chuẩn bị dữ liệu cho training"""
        
        # Cho PyTorch models
        dataset = PhoneSequenceDataset(phone_numbers, labels)
        
        # Cho TensorFlow models
        sequences = []
        features = []
        
        for phone in phone_numbers:
            # Sequence data
            clean_phone = re.sub(r'\D', '', phone)
            if len(clean_phone) > 15:
                clean_phone = clean_phone[:15]
            else:
                clean_phone = clean_phone.ljust(15, '0')
            
            sequence = [int(d) for d in clean_phone]
            sequences.append(sequence)
            
            # Feature data
            feature_vector = self._extract_features(phone)
            features.append(feature_vector)
        
        return {
            'pytorch_dataset': dataset,
            'tf_sequences': np.array(sequences),
            'tf_features': np.array(features),
            'labels': np.array(labels)
        }
    
    def _extract_features(self, phone_number: str) -> List[float]:
        """Trích xuất features cho dense network"""
        clean_number = re.sub(r'\D', '', phone_number)
        
        if not clean_number:
            return [0.0] * 50
        
        features = []
        
        # Basic features
        features.append(len(clean_number))
        features.append(len(set(clean_number)))
        features.append(int(clean_number.startswith('0')))
        features.append(int(clean_number.startswith('84')))
        
        # Digit counts
        for digit in range(10):
            features.append(clean_number.count(str(digit)))
        
        # Statistical features
        digits = [int(d) for d in clean_number]
        features.append(np.mean(digits))
        features.append(np.std(digits))
        features.append(np.var(digits))
        features.append(np.median(digits))
        
        # Pattern features
        features.append(self._max_consecutive_same(clean_number))
        features.append(int(self._is_palindrome(clean_number)))
        features.append(int(self._is_ascending(clean_number)))
        features.append(int(self._is_descending(clean_number)))
        
        # Frequency features
        features.append(sum(1 for d in clean_number if int(d) % 2 == 0))
        features.append(sum(1 for d in clean_number if int(d) % 2 == 1))
        
        # Ratios
        features.append(len(set(clean_number)) / len(clean_number))
        features.append(clean_number.count('0') / len(clean_number))
        
        # Prefix features
        if len(clean_number) >= 3:
            prefix = clean_number[:3]
            vn_prefixes = ['032', '033', '034', '035', '036', '037', '038', '039']
            features.append(int(prefix in vn_prefixes))
        else:
            features.append(0)
        
        # Pad hoặc truncate để có đúng 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return features[:50]
    
    def train_pytorch_models(self, data: Dict, epochs: int = 100, batch_size: int = 32):
        """Train PyTorch models"""
        dataset = data['pytorch_dataset']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for sequences, labels in dataloader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                if (epoch + 1) % 20 == 0:
                    accuracy = 100 * correct / total
                    logger.info(f'{name} - Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
    
    def train_tensorflow_models(self, data: Dict, epochs: int = 100, batch_size: int = 32):
        """Train TensorFlow models"""
        sequences = data['tf_sequences']
        features = data['tf_features']
        labels = data['labels']
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train dense model
        logger.info("Training dense model...")
        self.tf_models['dense'].fit(
            features, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train LSTM model
        logger.info("Training LSTM model...")
        self.tf_models['lstm_tf'].fit(
            sequences, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train CNN model
        logger.info("Training CNN model...")
        self.tf_models['cnn_tf'].fit(
            sequences, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
    
    def train_all_models(self, phone_numbers: List[str], labels: List[int], epochs: int = 100):
        """Train tất cả models"""
        logger.info("Preparing data...")
        data = self.prepare_data(phone_numbers, labels)
        
        logger.info("Training PyTorch models...")
        self.train_pytorch_models(data, epochs)
        
        logger.info("Training TensorFlow models...")
        self.train_tensorflow_models(data, epochs)
        
        self.is_trained = True
        logger.info("All models trained successfully!")
    
    def predict_single(self, phone_number: str) -> Dict:
        """Predict cho một số điện thoại"""
        if not self.is_trained:
            raise ValueError("Models chưa được train")
        
        predictions = {}
        
        # PyTorch predictions
        clean_phone = re.sub(r'\D', '', phone_number)
        if len(clean_phone) > 15:
            clean_phone = clean_phone[:15]
        else:
            clean_phone = clean_phone.ljust(15, '0')
        
        sequence = torch.tensor([int(d) for d in clean_phone], dtype=torch.long).unsqueeze(0).to(self.device)
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                outputs = model(sequence)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                
                predictions[name] = {
                    'prediction': int(predicted.item()),
                    'probability': float(probabilities[0, 1].item()),
                    'confidence': float(torch.max(probabilities).item())
                }
        
        # TensorFlow predictions
        features = np.array([self._extract_features(phone_number)])
        features = self.scaler.transform(features)
        
        sequences = np.array([[int(d) for d in clean_phone]])
        
        # Dense model
        dense_pred = self.tf_models['dense'].predict(features, verbose=0)
        predictions['dense'] = {
            'prediction': int(np.argmax(dense_pred[0])),
            'probability': float(dense_pred[0, 1]),
            'confidence': float(np.max(dense_pred[0]))
        }
        
        # LSTM model
        lstm_pred = self.tf_models['lstm_tf'].predict(sequences, verbose=0)
        predictions['lstm_tf'] = {
            'prediction': int(np.argmax(lstm_pred[0])),
            'probability': float(lstm_pred[0, 1]),
            'confidence': float(np.max(lstm_pred[0]))
        }
        
        # CNN model
        cnn_pred = self.tf_models['cnn_tf'].predict(sequences, verbose=0)
        predictions['cnn_tf'] = {
            'prediction': int(np.argmax(cnn_pred[0])),
            'probability': float(cnn_pred[0, 1]),
            'confidence': float(np.max(cnn_pred[0]))
        }
        
        # Ensemble prediction
        all_predictions = [p['prediction'] for p in predictions.values()]
        all_probabilities = [p['probability'] for p in predictions.values()]
        
        ensemble_prediction = int(np.mean(all_predictions) > 0.5)
        ensemble_probability = np.mean(all_probabilities)
        ensemble_confidence = np.mean([p['confidence'] for p in predictions.values()])
        
        return {
            'phone_number': phone_number,
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_prediction,
            'ensemble_probability': ensemble_probability,
            'ensemble_confidence': ensemble_confidence,
            'consensus_strength': abs(np.mean(all_predictions) - 0.5) * 2,
            'model_agreement': len(set(all_predictions)) == 1
        }
    
    def save_models(self, path: Optional[str] = None):
        """Lưu tất cả models"""
        if path is None:
            path = self.model_path
        
        os.makedirs(path, exist_ok=True)
        
        # Save PyTorch models
        torch_path = os.path.join(path, 'pytorch_models')
        os.makedirs(torch_path, exist_ok=True)
        
        for name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(torch_path, f'{name}.pth'))
        
        # Save TensorFlow models
        tf_path = os.path.join(path, 'tensorflow_models')
        os.makedirs(tf_path, exist_ok=True)
        
        for name, model in self.tf_models.items():
            model.save(os.path.join(tf_path, f'{name}.h5'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        logger.info(f"All models saved to {path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load tất cả models"""
        if path is None:
            path = self.model_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path {path} không tồn tại")
        
        # Load PyTorch models
        torch_path = os.path.join(path, 'pytorch_models')
        if os.path.exists(torch_path):
            for name, model in self.models.items():
                model_file = os.path.join(torch_path, f'{name}.pth')
                if os.path.exists(model_file):
                    model.load_state_dict(torch.load(model_file, map_location=self.device))
        
        # Load TensorFlow models
        tf_path = os.path.join(path, 'tensorflow_models')
        if os.path.exists(tf_path):
            for name in self.tf_models.keys():
                model_file = os.path.join(tf_path, f'{name}.h5')
                if os.path.exists(model_file):
                    self.tf_models[name] = tf.keras.models.load_model(model_file)
        
        # Load scaler
        scaler_file = os.path.join(path, 'scaler.pkl')
        if os.path.exists(scaler_file):
            self.scaler = joblib.load(scaler_file)
        
        self.is_trained = True
        logger.info(f"All models loaded from {path}")
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[List[str], List[int]]:
        """Tạo dữ liệu synthetic"""
        import random
        
        phone_numbers = []
        labels = []
        
        # Valid numbers
        vn_prefixes = ['032', '033', '034', '035', '036', '037', '038', '039',
                      '070', '071', '072', '073', '074', '075', '076', '077', '078', '079']
        
        for _ in range(n_samples // 2):
            prefix = random.choice(vn_prefixes)
            suffix = ''.join(random.choices('0123456789', k=7))
            phone = f"0{prefix}{suffix}"
            phone_numbers.append(phone)
            labels.append(1)
        
        # Invalid numbers
        for _ in range(n_samples // 2):
            pattern = random.choice(['repeated', 'sequential', 'random'])
            
            if pattern == 'repeated':
                digit = random.choice('0123456789')
                phone = digit * random.randint(8, 12)
            elif pattern == 'sequential':
                start = random.randint(0, 5)
                phone = ''.join(str((start + i) % 10) for i in range(10))
            else:
                phone = ''.join(random.choices('0123456789', k=random.randint(8, 12)))
            
            phone_numbers.append(phone)
            labels.append(0)
        
        return phone_numbers, labels
    
    # Helper methods
    def _max_consecutive_same(self, digits: str) -> int:
        """Đếm chữ số liên tiếp giống nhau tối đa"""
        if not digits:
            return 0
        
        max_count = 1
        current_count = 1
        
        for i in range(1, len(digits)):
            if digits[i] == digits[i-1]:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 1
        
        return max_count
    
    def _is_palindrome(self, digits: str) -> bool:
        """Kiểm tra palindrome"""
        return digits == digits[::-1] and len(digits) > 4
    
    def _is_ascending(self, digits: str) -> bool:
        """Kiểm tra dãy tăng dần"""
        if len(digits) < 4:
            return False
        
        for i in range(len(digits) - 3):
            if (int(digits[i+1]) == int(digits[i]) + 1 and
                int(digits[i+2]) == int(digits[i+1]) + 1 and
                int(digits[i+3]) == int(digits[i+2]) + 1):
                return True
        return False
    
    def _is_descending(self, digits: str) -> bool:
        """Kiểm tra dãy giảm dần"""
        if len(digits) < 4:
            return False
        
        for i in range(len(digits) - 3):
            if (int(digits[i+1]) == int(digits[i]) - 1 and
                int(digits[i+2]) == int(digits[i+1]) - 1 and
                int(digits[i+3]) == int(digits[i+2]) - 1):
                return True
        return False
