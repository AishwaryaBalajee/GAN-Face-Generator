import tensorflow as tf

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from PIL import Image

from IPython import display

dataset_dir = r"/home/ab2576/dl/dataset/img_align_celeba/"
no_of_images = 60000

train_images = []

for num, image in enumerate(os.listdir(dataset_dir)):
    if image.endswith('.jpg') and num <= no_of_images:
        if num%5000 == 0:
            print(num)
        img = Image.open(os.path.join(dataset_dir, image))  

        width, height = img.size
        img = img.crop((0, 20, 178, 198))

        img = img.resize((100, 100))

        img = np.array(img)
        train_images.append(img)
        # result.save('output.jpg') 

train_images = np.array(train_images)

# train_images = train_images.reshape(train_images.shape[0], 100, 100, 3).astype('float')
# train_images = (train_images - 127.5) / 127.5

def process_batch(batch):
    # Reshape and normalize the batch
    batch = batch.reshape(batch.shape[0], 100, 100, 3).astype('float32')
    batch = (batch - 127.5) / 127.5
    return batch

# Assuming train_images is a large numpy array
BATCH_SIZE = 500  # Adjust based on your system's memory

# Calculate the number of batches needed
num_batches = int(np.ceil(train_images.shape[0] / BATCH_SIZE))

# Process in batches
processed_batches = []
for i in range(num_batches):
    batch = train_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    processed_batch = process_batch(batch)
    processed_batches.append(processed_batch)
    # Optionally: Use the processed batch directly instead of storing

# If you really need to concatenate them back into a single array (warning: memory intensive!)
train_images = np.concatenate(processed_batches, axis=0)


BUFFER_SIZE = 500
BATCH_SIZE = 500
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((25, 25, 256)))
    assert model.output_shape == (None, 25, 25, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 100, 100, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[100, 100, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)

      image = predictions[i, :, :, :] * 127.5 + 127.5
      image = tf.cast(image, tf.uint8).numpy()
      plt.imshow(image)
      # plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5).astype('uint8'))
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))  

EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(dataset, epochs):
  gen_loss_arr = []
  disc_loss_arr = []
  for epoch in range(epochs):
    print(epoch, '/',  epochs)
    start = time.time()

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)
    
    gen_loss_arr.append(gen_loss)
    disc_loss_arr.append(disc_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(gen_loss_arr, label='Generator Loss')
    plt.plot(disc_loss_arr, label='Discriminator Loss')
    plt.title('Losses at Epoch {}'.format(epoch + 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'/home/ab2576/dl/loss_epoch_{epoch + 1}.png')
    plt.close()
    
    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
                           
train(train_dataset, EPOCHS)
  
