import nunmpy as np
import matplotlib.pyplot as plt
import json as js
import os

from matplotlib.animation import FuncAnimation
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model



class GAN:

    def __init__(self, params_json, params_filepath=None, load_params=False) -> None:
        
        
        self.params_json = params_json
        if load_params:
            self.load_params(filepath=params_filepath)
        
        self.dis_conv_layers_n = len(self.params_json["dis_conv_filters"])
        self.dis_dense_layers_n = len(self.params_json["dis_dense_units"])
        self.gen_conv_layer = len(self.params_json["gen_conv_filters"])
        self.weights_init = RandomNormal(mean=self.params_json["weights_init"]["mean"],stddev=self.params_json["weights_init"]["stddev"])


        self.gen_losses = []
        self.dis_losses = []
        self.epoch_iterator = 0

    def load_params(self, filepath):

        if filepath is None:
            filepath = "C:\\Users\\1\\Desktop\\PersonalFriendProject\\models_params\\GAN.json"
        with open(filepath, "r") as json_file:
            self.params_json = js.laod(json_file)

    def _build_discriminator_(self):

        input_layer = Input(shape=self.params_json["input_shape"][:-1])
        conv_layer = input_layer
        
        for layer_n in range(self.dis_conv_layers_n):

            conv_layer = Conv2D(filters=self.params_json["dis_conv_filters"][layer_n], 
                                kernel_size=self.params_json["dis_conv_kernel_size"][layer_n],
                                padding="same", kernel_initializer=self.weights_init,
                                strides=self.params_json["dis_conv_strides"][layer_n])(conv_layer)
            
            conv_layer = Activation(self.params_json["dis_conv_activation"][layer_n])(conv_layer)
            conv_layer = Dropout(rate=self.params_json["dis_conv_doprout_rates"][layer_n])(conv_layer)
            conv_layre = BatchNormalization()(conv_layer)
        
        self.saved_shape = conv_layer.shape[1:]
        dense_layer = Flatten()(conv_layer)

        for layer_n in range(self.dis_dense_layers_n):

            dense_layer = Dense(self.params_json["dis_dense_units"][layer_n],
                                activation=self.params_json["dis_dense_activations"][layer_n])(dense_layer)
            
            dense_layer = Dropout(rate=self.params_json["dis_dense_dropout_rates"][layer_n])
        
        output_layer = Dense(units=1, activation="sigmoid")(dense_layer)
        self.discriminator = Model(input_layer, output_layer)
    
    def set_trainable(self, model, value):

        model.trainable = value
        for layer in model.layers:
            layer.trainable = value

    def _build_generator_(self):

        input_layer = Input(shape=(self.params_json["noise_dim"], ))
        rec_layer = Dense(units=np.prod(self.saved_shape))(input_layer)
        rec_layer = Reshape(target_shape=self.saved_shape)(rec_layer)
        conv_layer = rec_layer

        for layer_n in range(self.gen_conv_layer):

            conv_layer = Conv2DTranspose(filters=self.params_json["gen_conv_filters"][layer_n], 
                                    kernel_size=self.params_json["gen_conv_kernel_size"][layer_n],
                                    padding="same", kernel_initializer=self.weights_init,
                                    strides=self.params_json["gen_conv_strides"][layer_n])(conv_layer)
            
            conv_layer = Activation(self.params_json["gen_conv_activation"][layer_n])(conv_layer)
            conv_layer = Dropout(rate=self.params_json["gen_conv_doprout_rates"][layer_n])(conv_layer)
            conv_layre = BatchNormalization()(conv_layer)
        
        output_layer = Conv2D(filters=self.params_json["input_shape"][-1], kernel_size=3, padding="same")(conv_layer)
        output_layer = Activation(self.params_json["gen_out_activation"])
        self.generator = Model(input_layer, output_layer)

    
    def _build_model_(self):

        self.discriminator.compile(
            optimizer=RMSprop(learning_rate=self.params_json["dis_learning_rate"]),
            loss=self.params_json["dis_loss_function"],
            metrics=self.params_json["dis_metrics"]
        )
        self.set_trainable(model=self.discriminator, value=False)

        model_input = Input(shape=(self.params_json["noise_dim"], ))
        model_output = self.discriminator(self.generator(model_input))
        
        self.model = Model(model_input, model_output)
        self.model.compile(
            optimizer=RMSprop(learning_rate=self.params_json["entire_model_learning_rate"]),
            loss=self.params_json["entire_model_loss_function"],
            metrics=self.params_json["entire_model_metric"]
        )

        self.set_trainable(model=self.discriminator, value=True)

    def _train_dis_(self, train_tensor, batch_size):

        random_idx = np.random.randint(0, train_tensor.shape[0] - 1, batch_size)
        valid_labels = np.ones(batch_size)
        fake_labels = np.zeros(batch_size)
        
        noise = np.random.normal(0, 0.12, (batch_size, self.params_json["noise_dim"]))
        true_samples = train_tensor[random_idx]
        gen_samples = self.generator.predict(noise)

        real_history = self.discriminator.train_on_batch(true_samples, valid_labels)
        fake_history = self.discriminator.train_on_batch(gen_samples, fake_labels)

        return [real_history, fake_history]

    def _train_gen_(self, train_tensor, batch_size):

        noise = np.random.normal(0, 0.12, (batch_size, self.params_json["noise_dim"]))
        valid_labels = np.ones(batch_size)
        return self.model.train_on_batch(noise, valid_labels)

    def save_samples(self, samples_number, gen_folder):
        
        curent_epoch_samples = os.path.join(gen_folder, f"generated_samples_{self.epoch_iterator}.png")

        samples_number_sq = int(np.sqrt(samples_number))
        fig, axis = plt.subplots()

        if not (self.params_json["input_shape"][-1] == 1):
            show_tensor = np.zeros((samples_number_sq * self.params_json["input_shape"][0], samples_number_sq * self.params_json["input_shape"][1], self.params_json["input_shape"][-1]))
        
        else:
            show_tensor = np.zeros((samples_number_sq * self.params_json["input_shape"][0], samples_number_sq * self.params_json["input_shape"][1]))

        noise = np.random.normal(0, 0.12, (samples_number, self.params_json["noise_dim"]))
        gen_samples = self.generator.predict(noise)

        sample_number = 0

        for i in range(samples_number_sq):
            for j in range(samples_number_sq):
                
                if not (self.params_json["input_shape"][-1] == 1):
                    show_tensor[i * self.params_json["input_shape"][0]: (i + 1) * self.params_json["input_shape"][0],
                                j * self.params_json["input_shape"][0]: (j + 1) * self.params_json["input_shape"][1], :] = gen_samples[sample_number]
                
                else:
                    show_tensor[i * self.params_json["input_shape"][0]: (i + 1) * self.params_json["input_shape"][0],
                                j * self.params_json["input_shape"][0]: (j + 1) * self.params_json["input_shape"][1]] = gen_samples[sample_number]

                sample_number += 1
        
        
        axis.imshow(show_tensor, cmap="inferno")
        fig.savefig(curent_epoch_samples)

    def train_model(self, epochs, batch_size, run_folder, epoch_per_save, train_tensor):

        gs_folder = os.path.join(run_folder, "gen_samples_folder")
        weights_folder = os.path.join(run_folder, "model_weights_folder")
        
        folders = [gs_folder, weights_folder]
        for folder in folders:
            
            if not os.path.exists(folder):
                os.mkdir(folder)
        
        for epoch in range(epochs):

            self.dis_losses.append(self._train_dis_(train_tensor=train_tensor, batch_size=batch_size))
            self.gen_losses.append(self._train_gen_(train_tensor=train_tensor, batch_size=batch_size))

            if epoch % epoch_per_save == 0:
                self.save_samples(gen_folder=gs_folder, samples_number=25)

            self.epoch_iterator += 1

        generator_weights_path = os.path.join(weights_folder, "generator_model_weights.weights.h5")
        discriminator_weights_path = os.path.join(weights_folder, "discriminator_model_weights.weights.h5")
        entire_model_weights_path = os.path.join(weights_folder, "entire_model_weights.weights.h5")

        self.generator.save_weights(filepath=generator_weights_path)
        self.discriminator.save_weights(filepath=discriminator_weights_path)
        self.model.save_weights(filepath=entire_model_weights_path)
    
    def load_weights(self, filepath):

        for ws_file in os.listdir(filepath):
            
            ws_path = os.path.join(filepath, ws_path)
            if "generator" in ws_file:
                self.generator.load_weights(ws_path)
            
            elif "discriminator" in ws_file:
                self.discriminator.load_weights(ws_path)
            
            elif "entire_model" in ws_file:
                self.model.load_weights(ws_path)

