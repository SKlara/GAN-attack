from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import keras

import matplotlib.pyplot as plt

import numpy as np

# Load the dataset
(X_train, y_train), (_, _) = mnist.load_data()

# Configure input
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

def get_dataset(digit):
    idx = []
    for i in range(len(y_train)):
        if (y_train[i] == digit):
            idx.append(i)
    idx = np.array(idx)
    data = X_train[idx]
    digit = y_train[idx]
    digit = digit.reshape(-1, 1)
    return (data, digit)

class Discriminator():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 4

        #optimizer = SGD(1e-3, 0.0, 1e-7)
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['sparse_categorical_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(4, activation='softmax'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, x_zero, y_zero, x_one, y_one, *args):

        batch_size = 64
        batch_per_digit = 36

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_zero.shape[0], batch_per_digit)
            imgs_0, labels_0 = x_zero[idx], y_zero[idx]

            if(len(args) > 0):
                gen_images = args[0]
                fake_labels = args[1]
                idx = np.random.randint(0, x_one.shape[0], 18)
                imgs_1, labels_1 = x_one[idx], y_one[idx]
                idx = np.random.randint(0, gen_images.shape[0], 18)
                imgs_g, labels_g = x_one[idx], y_one[idx]

                imgs = np.concatenate([imgs_0, imgs_1, imgs_g])
                labels = np.concatenate([labels_0, labels_1, labels_g])

            else:
                idx = np.random.randint(0, x_one.shape[0], batch_per_digit)
                imgs_1, labels_1 = x_one[idx], y_one[idx]

                imgs = np.concatenate([imgs_0, imgs_1])
                labels = np.concatenate([labels_0, labels_1])

            d_loss = self.discriminator.train_on_batch([imgs, labels], labels)

            print ("%d [D loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1]))

    def get_dscr_weight(self):
        return self.discriminator.get_weights()

    def set_dscr_weights(self, dscr):
        self.discriminator.set_weights(dscr)


class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 4
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['sparse_categorical_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        #optimizer = SGD(0.02)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['sparse_categorical_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(4, activation='softmax'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, x_zero, y_zero, x_one, y_one, n):

        batch_size = 63
        batch_per_digit = 21
        sample_interval = 10

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_zero.shape[0], batch_per_digit)
            imgs_0, labels_0 = x_zero[idx], y_zero[idx]
            idx = np.random.randint(0, x_one.shape[0], batch_per_digit)
            imgs_1, labels_1 = x_one[idx], y_one[idx]
            #idx = np.random.randint(0, x_two.shape[0], batch_per_digit)
            #imgs_2, labels_2 = x_two[idx], y_two[idx]
            
            imgs = np.concatenate([imgs_0, imgs_1])
            labels = np.concatenate([labels_0, labels_1])

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_per_digit, 100))

            # Generate a half batch of new images
            fake_labels = np.random.randint(0, 3, (batch_per_digit, 1)).reshape(-1, 1)
            #fake_labels = np.array([1, 1, 1])
            gen_imgs = self.generator.predict([noise, fake_labels])

            X = np.concatenate([imgs, gen_imgs])
            Y = np.concatenate([labels, fake_labels])
            gen_labels = []
            for i in range(batch_per_digit):
                gen_labels.append(3)
            gen_labels = np.array(gen_labels).reshape(-1, 1)
            real_lables = np.concatenate([labels, gen_labels])

            # Train the discriminator
            #d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            #d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss = self.discriminator.train_on_batch([X, Y], real_lables)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Condition on labels
            # elvileg csak 1-esek lesznek generalva
            sampled_labels = np.random.randint(1, 2, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], sampled_labels)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if n % sample_interval == 0:
                self.sample_images(n, epoch)

    def sample_images(self, n, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.random.randint(1, 2, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("attack_images/images_20/%d_%d.png" % (n, epoch))
        plt.close()

    def predict_samples(self, n):
        noise = np.random.normal(0, 1, (n, 100))
        fake_labels = np.ones((n, 1))
        gen_imgs = self.generator.predict([noise, fake_labels])
        return gen_imgs

    def set_model(self, dscr):
        self.discriminator.set_weights(dscr)


class Victim():
    def __init__(self, digit_0, digit_1):
        self.x_zero, self.y_zero = get_dataset(digit_0)
        self.x_one, self.y_one = get_dataset(digit_1)
        
        self.discriminator = Discriminator()

    def train(self, epochs):
        self.discriminator.train(epochs, self.x_zero, self.y_zero, self.x_one, self.y_one)

    def get_model(self):
        return self.discriminator.get_dscr_weight()

    def set_model(self, dscr):
        self.discriminator.set_dscr_weights(dscr)

class Adversary():
    def __init__(self, digit_0, digit_1):
        self.digit_1 = digit_1
        self.x_zero, self.y_zero = get_dataset(digit_0)
        self.x_one, self.y_one = get_dataset(digit_1)

        self.discriminator = Discriminator()
        self.cgan = CGAN()

    def train(self, epochs, n):
        self.cgan.train((epochs*5), self.x_zero, self.y_zero, self.x_one, self.y_one, n)
        self.discriminator.train(epochs, self.x_zero, self.y_zero, self.x_one, self.y_one)
        #gen_imgs = self.cgan.predict_samples(50)
        #fake_labels = []
        #for i in range(50):
        #    fake_labels.append(self.digit_1)
        #fake_labels = np.array(fake_labels).reshape(-1, 1)
        #self.discriminator.train(epochs, self.x_zero, self.y_zero, self.x_one, self.y_one, gen_imgs, fake_labels)

    def get_model(self):
        return self.discriminator.get_dscr_weight()

    def set_model(self, dscr):
        self.discriminator.set_dscr_weights(dscr)
        self.cgan.set_model(dscr)

if __name__ == '__main__':
    victim = Victim(digit_0=0, digit_1=1)
    adversary = Adversary(digit_0=0, digit_1=2)
    for i in range (10001):
        print ("epoch: %d" % i)
        victim.train(epochs=1)
        dscr = victim.get_model()
        adversary.set_model(dscr)
        adversary.train(epochs=1, n=i)
        dscr = adversary.get_model()
        victim.set_model(dscr)