from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time, psutil
import gc  # Add at top with other imports
import math

verbose = True # print logs

############## LOGGING ###############

def model_name_from_classifier_type(classifier_type):
    if classifier_type == 'linear':
        return "Logistic Regression"
    elif classifier_type == 'cnn':
        return "CNN"
    elif classifier_type == 'transformer':
        return "Transformer"
    else:
        raise ValueError(f"Invalid classifier type: {classifier_type}")

def log(message, priority=0, indent=0):
    max_log_priority = -1 if not verbose else 4
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}")


############## DATA PREPROCESSING ###############

from scipy import signal
import numpy as np
def compute_stft(data, fs=2048, preprocess="fft_abs"):
    """Compute spectrogram with both power and phase information for a single trial of data.
    
    Args:
        data (numpy.ndarray): Input voltage data of shape (n_channels, n_samples) or (batch_size, n_channels, n_samples)
        fs (int): Sampling frequency in Hz
        max_freq (int): Maximum frequency to include in Hz
    
    Returns:
        numpy.ndarray: Real-valued spectrogram representation containing both magnitude and phase information
                      Shape: (..., 2, n_freqs, n_times) where the 2 represents [magnitude, phase]
    """
    # For 1 second of data at 2048Hz, we'll use larger window
    #nperseg = 256  # 125 ms window
    #nperseg //= 2
    #noverlap = nperseg // 4 * 3 # 75% overlap
    noverlap = 0 # 0% overlap
    
    # Use STFT to get complex-valued coefficients
    f, t, Zxx = signal.stft(
        data,
        fs=fs, 
        nperseg=nperseg,
        noverlap=noverlap,
        window='boxcar'
    ) # Zxx shape: (n_channels, n_freqs, n_times)


    if preprocess == "fft_absangle":
        # Split complex values into magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        # Stack magnitude and phase along a new axis
        return np.stack([magnitude, phase], axis=-2)
    elif preprocess == "fft_realimag":
        real = np.real(Zxx)
        imag = np.imag(Zxx)
        return np.stack([real, imag], axis=-2)
    else:   
        magnitude = np.abs(Zxx)
        return magnitude
def downsample(data, fs=2048, downsample_rate=200):
    return signal.resample_poly(data, up=fs, down=downsample_rate, axis=-1)
def remove_line_noise(data, fs=2048, line_freq=60):
    """Remove line noise (60 Hz and harmonics) from neural data.
    
    Args:
        data (numpy.ndarray): Input voltage data of shape (n_channels, n_samples) or (batch_size, n_channels, n_samples)
        fs (int): Sampling frequency in Hz
        line_freq (int): Fundamental line frequency in Hz (typically 60 Hz in the US)
        
    Returns:
        numpy.ndarray: Filtered data with the same shape as input
    """
    # Make a copy of the data to avoid modifying the original
    filtered_data = data.copy()
    
    # Define the width of the notch filter (5 Hz on each side)
    bandwidth = 5.0
    
    # Calculate the quality factor Q
    Q = line_freq / bandwidth
    
    # Apply notch filters for the fundamental frequency and harmonics
    # We'll filter up to the 5th harmonic (60, 120, 180, 240, 300 Hz)
    for harmonic in range(1, 6):
        harmonic_freq = line_freq * harmonic
        
        # Skip if the harmonic frequency is above the Nyquist frequency
        if harmonic_freq > fs/2:
            break
            
        # Create and apply a notch filter
        b, a = signal.iirnotch(harmonic_freq, Q, fs)
        
        # Apply the filter along the time dimension
        if filtered_data.ndim == 2:  # (n_channels, n_samples)
            filtered_data = signal.filtfilt(b, a, filtered_data, axis=1)
        elif filtered_data.ndim == 3:  # (batch_size, n_channels, n_samples)
            for i in range(filtered_data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, filtered_data[i], axis=1)
    
    return filtered_data
def preprocess_data(data, preprocess):
    for preprocess_option in preprocess.split('-'):
        if preprocess_option in ['fft_absangle', 'fft_realimag', 'fft_abs']:
            data = compute_stft(data, preprocess=preprocess_option)
        elif preprocess_option == 'remove_line_noise':
            data = remove_line_noise(data)
        elif preprocess_option == 'downsample_200':
            data = downsample(data, downsample_rate=200)
    return data



############## CLASSIFICATION ###############


class TransformerClassifier:
    def __init__(self, random_state=42, max_iter=100, batch_size=128, learning_rate=0.001, val_size=0.2, tol=1e-4, patience=10,
                 d_model=64, nhead=8, dim_feedforward=256, dropout=0.1, num_layers=3):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_layers = num_layers

    def _create_model(self, input_shape, n_classes):
        class Transformer(torch.nn.Module):
            def __init__(self, input_shape, n_classes):
                super().__init__()
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    self.input_proj = torch.nn.Linear(input_shape[0], self.d_model)  # Project channels to embedding dim
                    self.pos_encoder = PositionalEncoding(self.d_model, max_len=input_shape[1])
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                    self.fc = torch.nn.Linear(self.d_model, n_classes)
                else:  # 3D input (channels, freq, time)
                    self.input_proj = torch.nn.Linear(input_shape[0] * input_shape[1], self.d_model)  # Project channels*freq to embedding dim
                    self.pos_encoder = PositionalEncoding(self.d_model, max_len=input_shape[2])
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                    self.fc = torch.nn.Linear(self.d_model, n_classes)
                
            def forward(self, x):
                # Reshape input for transformer
                if len(x.shape) == 3:  # (batch, channels, time)
                    x = x.transpose(1, 2)  # (batch, time, channels)
                    x = self.input_proj(x)  # (batch, time, 64)
                else:  # (batch, channels, freq, time)
                    batch_size, channels, freq, time = x.shape
                    x = x.transpose(1, 3)  # (batch, time, channels, freq)
                    x = x.reshape(batch_size, time, channels * freq)
                    x = self.input_proj(x)  # (batch, time, 64)
                
                # Add positional encoding
                x = self.pos_encoder(x)
                
                # Apply transformer
                x = self.transformer_encoder(x)
                
                # Global average pooling over time dimension
                x = x.mean(dim=1)
                
                # Final classification layer
                x = self.fc(x)
                return x
        
        class PositionalEncoding(torch.nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        
        return Transformer(input_shape, n_classes)
    
    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create train/val split
        indices = np.random.permutation(len(X))
        val_size = int(self.val_size * len(X))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                log(f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                        break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class CNNClassifier:
    def __init__(self, random_state=42, max_iter=100, batch_size=128, learning_rate=0.0001, val_size=0.2, tol=1e-4, patience=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.val_size = val_size
        self.tol = tol
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None
        self.best_val_auroc = 0.0
        
    def _create_model(self, input_shape, n_classes):
        class CNN(torch.nn.Module):
            def __init__(self, input_shape, n_classes):
                super().__init__()
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    self.conv1 = torch.nn.Conv1d(input_shape[0], 32, kernel_size=3, padding=1)
                    self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool1d(2)
                    self.dropout = torch.nn.Dropout(0.5)
                    
                    # Calculate the size after convolutions and pooling
                    conv_output_size = input_shape[1] // 8 * 128
                    
                    self.fc1 = torch.nn.Linear(conv_output_size, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)
                    
                else:  # 3D input (channels, freq, time)
                    self.conv1 = torch.nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
                    self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.pool = torch.nn.MaxPool2d(2)
                    self.dropout = torch.nn.Dropout(0.5)
                    
                    # Calculate the size after convolutions and pooling
                    conv_output_size = (input_shape[1] // 8) * (input_shape[2] // 8) * 128
                    
                    self.fc1 = torch.nn.Linear(conv_output_size, 256)
                    self.fc2 = torch.nn.Linear(256, n_classes)
                
                self.relu = torch.nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return CNN(input_shape, n_classes)
    
    def fit(self, X, y):
        # Convert to torch tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create train/val split
        indices = np.random.permutation(len(X))
        val_size = int(self.val_size * len(X))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        log(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples", priority=3, indent=2)
        
        # Create model
        input_shape = X.shape[1:]
        self.model = self._create_model(input_shape, n_classes)
        self.model = self.model.to(self.device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.max_iter):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size].to(self.device)
                batch_y = y_train[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss_value = criterion(val_outputs, y_val.to(self.device))
                val_loss = val_loss_value.item()
                
                # Calculate validation AUROC
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                # Convert to one-hot encoding for AUROC calculation
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    y_val_onehot[i, label] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                log(f"Epoch {epoch+1}/{self.max_iter}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val AUROC: {val_auroc:.4f}", priority=3, indent=2)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + self.tol:
                    best_val_auroc = val_auroc
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    log(f"New best model saved with val AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log(f"Early stopping triggered after {epoch+1} epochs", priority=3, indent=2)
                        break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        log(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}", priority=3, indent=2)
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)



############## REGION AVERAGING (FOR DS/DM SPLITS) ###############

def get_region_labels(subject):
    """
    subject: BrainTreebankSubject
    returns: np.ndarray of shape (n_channels,)
    """
    return subject.get_all_electrode_metadata()['DesikanKilliany'].to_numpy()

def combine_regions(X_train, X_test, regions_train, regions_test):
    """
    X_train: np.ndarray of shape (n_samples, n_channels_train, n_timebins, d_model) or (n_samples, n_channels_train, n_timesamples)
    X_test: np.ndarray of shape (n_samples, n_channels_test, n_timebins, d_model) or (n_samples, n_channels_test, n_timesamples)
    regions_train: np.ndarray of shape (n_channels_train,)
    regions_test: np.ndarray of shape (n_channels_test,)
    """
    # Find the intersection of regions between train and test
    unique_regions_train = np.unique(regions_train)
    unique_regions_test = np.unique(regions_test)
    common_regions = np.intersect1d(unique_regions_train, unique_regions_test)
    
    if X_train.ndim == 3:
        # Add a dummy dimension to X_train and X_test for d_model=1
        X_train = X_train[:, :, :, np.newaxis]
        X_test = X_test[:, :, :, np.newaxis]

    n_samples_train, _, n_timebins, d_model = X_train.shape
    n_samples_test = X_test.shape[0]
    n_regions_intersect = len(common_regions)
    
    # Create new arrays to store region-averaged data
    X_train_regions = np.zeros((n_samples_train, n_regions_intersect, n_timebins, d_model), dtype=X_train.dtype)
    X_test_regions = np.zeros((n_samples_test, n_regions_intersect, n_timebins, d_model), dtype=X_test.dtype)
    
    # For each common region, average across all channels with that region label
    for i, region in enumerate(common_regions):
        # Find channels corresponding to this region
        train_mask = regions_train == region
        test_mask = regions_test == region
        
        # Average across channels with the same region
        X_train_regions[:, i, :, :] = X_train[:, train_mask, :, :].mean(axis=1)
        X_test_regions[:, i, :, :] = X_test[:, test_mask, :, :].mean(axis=1)
    
    return X_train_regions, X_test_regions, common_regions