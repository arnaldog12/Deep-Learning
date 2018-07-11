import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Part 1 - Building the CNN
clf = Sequential()
clf.add(Convolution2D(filters=32, kernel_size=[3,3], activation='relu', input_shape=(64,64,3)))
clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Convolution2D(filters=32, kernel_size=[3,3], activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))

clf.add(Flatten())
clf.add(Dense(units=128, activation='relu'))
clf.add(Dense(units=1, activation='sigmoid'))

clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set', 
                                              target_size=(64,64), 
                                              batch_size=32, 
                                              class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

# the configuration below was enough to achieve accuracy higher than 80% in test set (80.70%)
clf.fit_generator(generator=train_set, 
                  steps_per_epoch=800,
                  epochs=5,
                  validation_data=test_set,
                  validation_steps=200)

# Part 3 - Making new predictions
def load_image_as_nparray(path, target_size=(64,64)):
    sample = image.load_img(path, target_size=target_size)
    sample = image.img_to_array(sample)
    return np.expand_dims(sample, axis=0)

dog_sample = load_image_as_nparray('dataset/single_prediction/cat_or_dog_1.jpg')
cat_sample = load_image_as_nparray('dataset/single_prediction/cat_or_dog_2.jpg')

print(train_set.class_indices)
print("dog_sample: cat or dog? {}".format(clf.predict(dog_sample)))
print("cat_sample: cat or dog? {}".format(clf.predict(cat_sample)))
