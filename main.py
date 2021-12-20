import streamlit as st
import pickle

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

import matplotlib.pyplot as plt

# Basic Config

st.set_page_config(layout='wide', initial_sidebar_state='collapsed', page_title="Tuesday ML")

regressor_type = "<class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>"
classifier_type = "<class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>"

# Header and info

st.header("Tuesday ML (BETA)")
with st.expander("About"):
	st.write("This is an app designed to make machine learning easier and more accessible. So far, you can only " +
			"train multilayer perceptron neural networks, but we will add more later. Sample data is available below " +
			" and sample models and model uploading will be available soon. Please keep in mind this is a " +
			"beta release and is still in development. This app was make with and is hosted on Streamlit, a " +
			"library that makes it easier to display web user interfaces and data with python.")
	st.download_button("Sample data", open('Classification data.csv'), file_name="Classification data.csv")

st.markdown("---")

def to_data(args):
	man_type = "Auto"
	if 'data_man_type' in st.session_state:
		man_type = st.session_state['data_man_type']
	if man_type == "Auto":
		out = {
				"X": np.array(file[file.columns[0]].tolist()).astype("float64").reshape((-1, 1)).tolist(),
				"y": np.array(file[file.columns[len(file.columns)-1]].tolist()).astype("float64").reshape((-1, 1)).tolist()
			}
		return out
	elif man_type == "History":
		pass
	elif man_type == "Multiple":
		out = {
			"X": np.array(file[args[0]]).astype("float64").tolist(),
			"y": np.array(file[args[1]]).astype("float64").tolist()
		}
		return out

def model_inputter(container, model_):
	container.subheader("Model")
	with container:
		input_, output_ = st.columns([1, 3])
		with input_:
			with st.expander("Inputs:", True):
				input_arr_ = [st.number_input("Input "+str(i_)) for i_ in range(model_.n_features_in_)]
		with output_:
			with st.expander("Output:", True):
				st.write(model_.predict(np.array(input_arr_).reshape((1, -1))))

# Data

data_is_ready = False

data = []

st.header("Data")
data_col1, data_col2 = st.columns([1, 2])
with data_col1:
	st.subheader("Upload data or model")
	raw_file = st.file_uploader("", ["csv", "joblib"])
	if raw_file is not None:
		if raw_file.name.endswith(".csv"):
			file = pd.read_csv(raw_file)
			with st.expander('Raw Data Preview'):
				shown_rows = st.slider("Shown rows: ", min(len(file), 5), len(file))
				st.table(file.head(shown_rows))
			if len(file.columns) < 2:
				st.error("Data must have at least 2 columns")
				st.stop()
if raw_file is not None and raw_file.name.endswith(".joblib"):
	model = pickle.load(raw_file)
	model_inputter(st.container(), model)
	st.stop()
with data_col2:
	st.subheader("Edit data")
	
	if raw_file is not None:
		data_man = st.selectbox("Data Reshaping: ", ["Auto", "Multiple"])
		st.session_state['data_man_type'] = data_man
		if data_man == "Auto":
			data_is_ready = True
			data = pd.DataFrame(to_data([]))
		if data_man == "History":
			# data_is_ready = True
			st.write("This is not done yet, check back later")
		elif data_man == "Multiple":
			x_cols = st.multiselect("Select input columns: ", file.columns)
			
			avail_y_cols = list(file.columns.copy())
			for taken in x_cols:
				avail_y_cols.remove(taken)
			
			y_cols = st.multiselect("Select output columns: ", avail_y_cols)
			
			def sharing():
				for x_col in x_cols:
					if x_col in y_cols:
						return True
				return False
			
			if len(x_cols) == 0:
				st.warning("Select at least 1 input column")
			elif len(y_cols) == 0:
				st.warning("Select at least 1 output column")
			elif sharing():
				st.error("X and Y cannot share columns")
			else:
				data_is_ready = True
				try:
					data = pd.DataFrame(to_data([x_cols, y_cols]))
				except:
					st.error("Error while parsing data")
					st.stop()
	else:
		st.write("Waiting for file...")
	if data_is_ready:
		with st.expander("Reshaped Output", False):
			st.write(data)

if not data_is_ready:
	st.stop()

st.markdown("---")

# Design

st.header("Design")

with st.expander("Data Split"):
	train_percent = st.slider("Training Data Proportion: ", 0.0, 1.0, 0.8, 0.01)
	st.write("Train Length: "+str(int(np.floor(train_percent*len(data["X"])))))
	st.write("Test Length: "+str(len(data["X"])-int(np.floor(train_percent*len(data["X"])))))

with st.expander("Configure Layers"):
	design_col1, design_col2 = st.columns([1, 2])
	with design_col1:
		num_layers = st.slider("Total Layers: ", 2, 20, value=5)

	with design_col2:
		layers = []
		for i in range(num_layers):
			if i == 0:
				st.write("Input Nodes: " + str(len(data["X"][0])))
			elif i == len(range(num_layers))-1:
				st.write("Output Nodes: "+str(len(data["y"][0])))
			else:
				layers.append(st.slider("Layer "+str(i)+" Nodes: ", 1, 10, value=5))

if st.checkbox("Show Advanced"):
	with st.expander("Advanced", True):
		activation = st.selectbox("Activation Function", ["identity", "logistic", "tanh", "relu"], index=3)
		solver_ = st.selectbox("Solver", ["Limited-memory BFGS", "Stochastic Gradient Descent", "SGD with ADAM"], index=2)
		solver_conversion = {
			"Limited-memory BFGS": "lbfgs",
			"Stochastic Gradient Descent": "sgd",
			"SGD with ADAM": "adam"
		}
		solver = solver_conversion[solver_]
		epochs = st.slider("Max-Epochs", 100, 1000000, value=100000)
		regularization = st.slider("L2 Regularization Term", 0.0001, 0.01, 0.0001, 0.0001, format="%e")

else:
	epochs = 100000
	activation = "relu"
	solver = "sgd"
	regularization = 0.0001

model_type = st.radio("Model type: ", ["Regression", "Classification"])

def checkbox_change():
	if design_is_ready:
		st.session_state['train_iter'] = str(int(st.session_state['train_iter']) + 1)
		get_model(st.session_state['train_iter'])

design_is_ready = st.checkbox("Ready for training", on_change=checkbox_change)

if not design_is_ready:
	st.stop()

st.markdown("---")

# Train

default_args = {
	"hidden_layers": layers,
	"alpha": regularization,
	"learning_rate_init": 0.001
}

def generate_model(hyper_params):
	if model_type == "Regression":
		return MLPRegressor(**hyper_params)
	elif model_type == "Classification":
		return MLPClassifier(**hyper_params)

def search(hyper_params_list, iters):
	best_hyper_params = {}
	best_loss = 9999999999999999999999999
	for hyper_params in hyper_params_list:
		average_loss = 0
		for _i_ in range(iters):
			local_model = generate_model(hyper_params=hyper_params)
			local_model.fit(X_train, y_train)
			average_loss += local_model.best_loss_
		average_loss = average_loss/iters
		if average_loss <= best_loss:
			best_loss = average_loss
			best_hyper_params = hyper_params
	return best_hyper_params

st.header("Train & Validate")

loading_space = st.empty()

def deep_np_array(arr):
	out_1 = np.array(arr)
	out_2 = []
	for j in range(len(out_1)):
		out_2.append(np.array(out_1[j]))
	return np.array(out_2)

@st.experimental_singleton
def get_data(iter_val):
	global X_train, X_test, y_train, y_test
	X_train, X_test, y_train, y_test = train_test_split(deep_np_array(data["X"]), deep_np_array(data["y"]), test_size=(1 - train_percent))
	return X_train, X_test, y_train, y_test

if 'train_iter' not in st.session_state:
	st.session_state['train_iter'] = "0"

if 'from_saved' not in st.session_state:
	st.session_state['from_saved'] = "0"

X_train, X_test, y_train, y_test = get_data(st.session_state['train_iter'])

y_train = y_train.reshape(-1)

losses = []

roc_curves = []

def train_button():
	if 'train_iter' in st.session_state:
		st.session_state['train_iter'] = str(int(st.session_state['train_iter'])+1)
	else:
		st.session_state['train_iter'] = "0"
	get_model(st.session_state['train_iter'])

@st.experimental_memo(max_entries=1)
def get_model(iter_val):
	print(iter_val)
	if st.session_state['from_saved'] == "1":
		return model
	if model_type == "Regression":
		local_model = MLPRegressor(hidden_layer_sizes=layers, max_iter=epochs, activation=activation, solver=solver, alpha=regularization)
	else:
		local_model = MLPClassifier(hidden_layer_sizes=layers, max_iter=epochs, activation=activation, solver=solver, alpha=regularization)
	try:
		local_model.fit(X=X_train, y=y_train)
	except ValueError:
		if model_type == "Classification":
			loading_space.error("Error while training: Labels may be continuous. Check labels and model type.")
			st.stop()
		else:
			loading_space.error("Error while training")
			st.stop()
	except:
		loading_space.error("Error while training")
		st.stop()
	return local_model

@st.experimental_memo(max_entries=1)
def get_losses(iter_val):
	return losses

def calc_roc_curve(pos_val_i_, y_pred_proba_):
	thresholds = np.divide(np.arange(0, 100), 100)
	false_pos_rates = [0, 1]
	true_pos_rates = [0, 1]
	true_pos_i = [i for i, item in enumerate(y_test) if item == pos_val]
	for thres in thresholds:
		true_pos_count = 0
		false_pos_count = 0
		for i, pred_proba in enumerate(y_pred_proba_):
			if pred_proba[pos_val_i_] >= thres:
				if pos_val == y_test[i][0] and i in true_pos_i:
					true_pos_count += 1
				else:
					false_pos_count += 1
		if true_pos_count + false_pos_count == 0:
			continue
		true_pos_rates.append(true_pos_count / len(true_pos_i))
		false_pos_rates.append(false_pos_count / (len(y_test) - len(true_pos_i)))
	
	pos_rates = [[false_pos_rates[i], true_pos_rates[i]] for i in range(len(true_pos_rates))]
	
	def sort_pos_rates(arr):
		return arr[0]
	
	pos_rates.sort(key=sort_pos_rates)
	
	pos_rates = [[i[0] for i in pos_rates], [i[1] for i in pos_rates]]
	
	return pos_rates

def calc_best_iter_model(iters):
	global model
	global losses
	best_model = 0
	best_metric = 99999
	loss_arr = []
	with loading_space.container():
		st.subheader("Training "+str(iters)+" iterations...")
		prog_bar = st.progress(0.0)
		if st.button("Stop Training"):
			st.session_state['train_iter'] = str(int(st.session_state['train_iter']) + 1)
			st.session_state['from_saved'] = "1"
			model = best_model
			get_model(st.session_state['train_iter'])
			st.session_state['from_saved'] = "0"
			return
	
	for j in range(iters):
		if model_type == "Regression":
			local_model = MLPRegressor(hidden_layer_sizes=layers, max_iter=100000)
		else:
			local_model = MLPClassifier(hidden_layer_sizes=layers, max_iter=100000)
		local_model.fit(X_train, y_train)
		loss_arr.append(np.array(local_model.loss_curve_))
		_y_pred = local_model.predict(X_test)
		_y_pred_proba = local_model.predict_proba(X_test)
		if local_model.best_loss_ < best_metric:
			best_metric = local_model.best_loss_
			best_model = local_model
		prog_bar.progress((j+1)/iters)
	st.session_state['train_iter'] = str(int(st.session_state['train_iter']) + 1)
	st.session_state['from_saved'] = "1"
	model = best_model
	get_model(st.session_state['train_iter'])
	st.session_state['from_saved'] = "0"
	
	max_loss_len = 0
	for loss_ in loss_arr:
		if len(loss_) > max_loss_len:
			max_loss_len = len(loss_)
	
	new_loss_arr = []
	
	def np_to_shape(arr, target_len):
		arr = arr.tolist()
		for unused_var in range(target_len - len(arr)):
			arr.append(0)
		return np.array(arr)
	
	for loss_ in loss_arr:
		new_loss_arr.append(np_to_shape(loss_, max_loss_len))
	
	losses = np.mean(new_loss_arr, axis=0)
	get_losses(st.session_state['train_iter'])
	return "bruh"

try:
	model = get_model(st.session_state['train_iter'])
	
	y_train_pred = model.predict(X_train)
	y_pred = model.predict(X_test)
except:
	st.error("Error while loading model")
	
	def try_again():
		st.session_state['train_iter'] = str(int(st.session_state['train_iter']) + 1)
		get_model(st.session_state['train_iter'])

	st.button("Try again", on_click=try_again)
	st.stop()

model_col1, model_col2 = st.columns([1, 2])

true_model_type = str(type(model))

with model_col1:
	st.subheader("Training")
	with st.expander("Single iteration", True):
		st.button("Re-train", on_click=train_button)
	
	with st.expander("Best from iterations", True):
		train_iters = st.slider("Iterations: ", 5, 200, 50, 5)
		st.button("Start", on_click=calc_best_iter_model, args=(train_iters, ))
	
	hyperparam_tuning = st.checkbox("Enable Hyperparameter Tuning")
	
	if hyperparam_tuning:
		with st.expander("Hyperparameter Tuning", True):
			st.write("Tuned hyperparameters may optionally be applied to your model")
			sweep_iters = st.slider("Sweep Iterations", 5, 100, value=50)
			possible_hyperparams = ["Model Depth", "Layer Size", "Regularization", "Learning Rate"]
			need_tuning_hyperparams = {}
			for possible_hyperparam in possible_hyperparams:
				need_tuning_hyperparams[possible_hyperparam] = st.checkbox(possible_hyperparam)
			if st.button("Run Sweep"):
				st.info("Coming soon")

with model_col2:
	st.subheader("Evaluation")
	with st.expander("Metrics", True):
		st.write("Epochs to converge: ", model.n_iter_)
		try:
			if true_model_type == regressor_type:
				st.write("Train MSE: ", mean_squared_error(y_train, y_train_pred))
				st.write("Train MAE: ", mean_absolute_error(y_train, y_train_pred))
				st.write("Test MSE: ", mean_squared_error(y_test, y_pred))
				st.write("Test MAE: ", mean_absolute_error(y_test, y_pred))
			elif true_model_type == classifier_type:
				y_pred_proba = model.predict_proba(X_test)
				st.write("Train Accuracy: ", accuracy_score(y_train, y_train_pred))
				st.write("Test Accuracy: ", accuracy_score(y_test, y_pred))
				st.write("ROC AUC: ", roc_auc_score(y_test, y_pred_proba, multi_class="ovr"))
		except:
			st.error("Error while evaluating metrics")
	with st.expander("Graphs", True):
		try:
			if true_model_type == classifier_type:
				allowed_graphs = ["Loss", "ROC"]
			else:
				allowed_graphs = ["Loss"]
			graph_type = st.selectbox("Graph type: ", allowed_graphs)
			if graph_type == "ROC":
				pos_val = st.selectbox("Positive Value: ", model.classes_)
				pos_val_i = model.classes_.tolist().index(pos_val)
				pos_rates = calc_roc_curve(pos_val_i, y_pred_proba)
				fig, ax = plt.subplots()
				ax.plot(pos_rates[0], pos_rates[1], label="Model")
				ax.plot([0, 1], [0, 1], label="Random Cls")
				ax.legend()
				st.pyplot(fig)
			else:
				fig, ax = plt.subplots()
				ax.plot(model.loss_curve_, label='Model')
				losses = get_losses(st.session_state['train_iter'])
				if len(losses) > 0:
					ax.plot(losses, label='Average')
				ax.legend()
				st.pyplot(fig)
		except:
			st.error("Error while rendering graphs")

st.markdown("---")

# Export

st.header("Export")

st.download_button("Download Model", data=pickle.dumps(model), file_name="model.joblib")

model_inputter(st.container(), model)
