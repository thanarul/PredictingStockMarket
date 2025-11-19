from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def train_lstm_classifier(df):
	'''
	LSTM Classifier for predicting next-day direction
	'''

	df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
	df = df.dropna().reset_index(drop=True)

	# 2 Feature matrix 
	feature_cols = [
		"Close", "High", "Low", "Open", "Volume",
		"SMA", "EMA", "RSI", "MACD", "Signal", "Vol_Change"
	]
	X = df[feature_cols].values 
	y = df["Target"].values	


	# Scale Features 
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# Building sequences for lstm 
	seq_len = 30 # last 30 days as input window 

	def make_sequences(X_arr, y_arr, window):
		X_seq, y_seq = [], []
		for i in range(window - 1, len(X_arr)):
			X_seq.append(X_arr[i - window + 1 : i + 1])
			y_seq.append(y_arr[i])
		return np.array(X_seq), np.array(y_seq)

	X_seq, y_seq = make_sequences(X_scaled, y, seq_len)

	# time based train-test split
	split_idx = int(0.8 * len(X_seq))
	X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
	y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

	# 4 Define LSTM model 
	n_timesteps = X_train.shape[1]
	n_features = X_train.shape[2]

	model = Sequential([
		tf.keras.Input(shape=(n_timesteps, n_features)),
        LSTM(64, input_shape=(n_timesteps, n_features)),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])

	model.compile(
		loss="binary_crossentropy",
		optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
	)

	es = EarlyStopping(
		monitor = "val_loss",
		patience = 5,
		restore_best_weights = True,
		verbose = 1
	)

	# 5 Train the model 

	history = model.fit(
		X_train, y_train,
		validation_split = 0.1,
		epochs = 20,
		batch_size = 32,
		callbacks = [es],
		verbose = 1
	)

	#6 Evaluate 

	probs = model.predict(X_test)
	preds = (probs > 0.5).astype(int)

	print("\n📈 LSTM — FINAL TEST PERFORMANCE")
	print("Test Accuracy:", accuracy_score(y_test, preds))
	print(classification_report(y_test, preds))

	return model

