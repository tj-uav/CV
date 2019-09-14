import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import Model
from scipy import interpolate
from datetime import datetime
import os, sys, time

# Hyperparameters
EPOCHS = 100
LEARNING_RATE = 1e-04
INPUT_DIM = 5
TRAIN_FRAC = 0.75
NUM_SAMPLES = 10001
BATCH_SIZE = 64
STEP_SIZE = 10
SHOW = False
SAVE_MODEL = False

class SineModel(Model):
    def __init__(self, *args, **kwargs):
        super(SineModel, self).__init__(name='sine_model')
        self.d1 = tf.keras.layers.Dense(64, input_shape=(INPUT_DIM, None),
            kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')
        self.d2 = tf.keras.layers.Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')
        # self.d3 = tf.keras.layers.Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu')
        self.d4 = tf.keras.layers.Dense(1, kernel_initializer='random_uniform', bias_initializer='zeros', activation=None)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        # x = self.d3(x)
        return self.d4(x)

@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        prediction = model(features)
        loss = loss_object(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return prediction

@tf.function
def test_step(features, labels):
    prediction = model(features)
    loss = loss_object(labels, prediction)
    test_loss(loss)
    return prediction

def create_data():
    x_vals = np.linspace(0, 30*np.pi, NUM_SAMPLES)
    data = np.zeros((1, INPUT_DIM+1))
    for start_point in range(NUM_SAMPLES-(INPUT_DIM*STEP_SIZE)):
        row = [np.sin(x_vals[x]) for x in range(start_point, start_point+(INPUT_DIM+1)*STEP_SIZE, STEP_SIZE)]
        data = np.concatenate([data, np.reshape(row, (1, INPUT_DIM+1))], axis=0)
    data = np.delete(data, 0, axis=0)  # Remove first row of zeros
    return data

def form_results():
    folder_name = "./{0}@{1}_Sine_Model". \
        format(datetime.today().strftime('%Y-%m-%d'), datetime.now().time().strftime('%H%M'))
    tensorboard_path = folder_name + '/Tensorboard'
    saved_model_path = folder_name + '/Saved_models'
    checkpoint_path = folder_name + '/Checkpoints'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        os.makedirs(tensorboard_path)
        os.makedirs(saved_model_path)
        os.makedirs(checkpoint_path)
    return tensorboard_path, saved_model_path, checkpoint_path

if __name__ == "__main__":
    # Load data and create datasets
    data = create_data()
    data_size = np.size(data, axis=0)
    test_indices = np.random.randint(0, data_size, size=int((1-TRAIN_FRAC)*NUM_SAMPLES))
    all_indices = np.arange(data_size)
    all_indices = np.delete(all_indices, test_indices)  # Delete overlap of indices saved for testing
    train_indices = np.random.choice(all_indices, size=int(TRAIN_FRAC*NUM_SAMPLES))
    train_data = data[train_indices,:]
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_data[:,:INPUT_DIM], train_data[:,INPUT_DIM:]))
    test_data = data[test_indices,:]
    test_dataset = tf.data.Dataset.from_tensor_slices((
        test_data[:,:INPUT_DIM], test_data[:,INPUT_DIM:]))

    # Organize data
    train_dataset.batch(BATCH_SIZE, drop_remainder=True).shuffle(10000)
    test_dataset.batch(BATCH_SIZE, drop_remainder=True).shuffle(10000)

    # Initialize model
    model = SineModel()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    print('Model successfully initialized.')

    # Define metrics to record in TensorBoard
    if SAVE_MODEL:
        tensorboard_path, saved_model_path, checkpoint_path = form_results()
        train_log_dir = tensorboard_path + '/Training'
        test_log_dir = tensorboard_path + '/Testing'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Train and evaluate the model
    print('Beginning training.')
    for epoch in range(EPOCHS):
        show = SHOW
        for features, labels in train_dataset:
            prediction = train_step(np.reshape(features, (1, len(features))), labels)
            if SAVE_MODEL:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=epoch)
                    tf.summary.histogram('d1_weights', model.d1.get_weights()[0], step=epoch)
                    # tf.summary.histogram('d2_weights', model.d2.get_weights()[0], step=epoch)
                    # tf.summary.histogram('d3_weights', model.d3.get_weights()[0], step=epoch)
        for features, labels in test_dataset:
            prediction = test_step(np.reshape(features, (1, len(features))), labels)
            if SAVE_MODEL:
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss.result(), step=epoch)
            if show:
                points = np.zeros((1,2))
                predict_points = np.zeros((1,2))
                for x in features:
                    points = np.concatenate([points, np.reshape(np.array([np.arcsin(x), x]), (1,2))], axis=0)
                    predict_points = np.concatenate([predict_points, np.reshape(np.array([np.arcsin(x), x]), (1,2))], axis=0)
                points = np.concatenate([points, np.reshape(np.array([np.arcsin(labels), labels]), (1,2))], axis=0)
                predict_points = np.concatenate([predict_points, np.reshape(np.array([np.arcsin(labels), prediction.numpy()]), (1,2))], axis=0)
                
                ax = plt.axes()
                plt.ion()
                plt.show()
                points_x = np.asarray(points[1:,0])
                points_y = np.asarray(points[1:,1])
                tck, u = interpolate.splprep([points_x, points_y], s=0)
                x_new, y_new = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
                spline_1 = plt.plot(points_x, points_y, x_new, y_new, label='Given spline', alpha=0.5, c='b')
                plt.draw()
                plt.pause(0.001)

                points_x = np.asarray(predict_points[1:,0], dtype=np.float64)
                points_y = np.asarray(predict_points[1:,1], dtype=np.float64)
                tck, u = interpolate.splprep([points_x, points_y], s=0)
                x_new, y_new = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
                spline_2 = plt.plot(points_x, points_y, x_new, y_new, label='Predicted spline', alpha=0.5, c='r')
                plt.draw()
                plt.pause(0.001)

                plt.axis('equal')
                if epoch == 0 and show == True:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc='upper left')
                show = False
        if epoch % 1 == 0 or epoch == EPOCHS-1:
            template = 'Epoch {}, Loss: {}, Test Loss: {}'
            if epoch == EPOCHS-1:
                print(template.format(epoch+1, train_loss.result(), test_loss.result()))
            else:
                print(template.format(epoch, train_loss.result(), test_loss.result()))

        # Reset metrics every epoch and shuffle the datasets
        train_loss.reset_states()
        test_loss.reset_states()
        train_dataset = train_dataset.shuffle(20000)
        test_dataset = test_dataset.shuffle(20000)
    
    if SAVE_MODEL:
        tf.saved_model.save(model, saved_model_path)
        print('Model was saved successfully to {}'.format(saved_model_path))
