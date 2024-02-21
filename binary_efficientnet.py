import cv2
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy

augmentation = False

def augment(img):
    #add any image augmentation here
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1) #horizontal flip, example add more with seperate probs
    #if np.random.rand() < 0.2:
    return img

def efficientnet_binary(input_shape=(224, 224, 3), num_classes=1):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape, classes=num_classes)
    #may set weights to false, imagenet for transfer learning

    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation='sigmoid')(x) # change head to binary

    model = Model(inputs=base.input, outputs=output)

    return model


def load_images_and_labels(directory, img_size=(224, 224)):
    #set up for a binary folder structure (e.g. train(folder) -> 0 (sub-folder), 1 (sub-folder))
    images = []
    labels = []

    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            
            # Load image with cv2
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            img = img / 255.0  # Normalization is important

            if augmentation==True:
                if np.random.rand() < 0.5: #augmentation chance, do for half images
                    aug_img = augment(img)
                    images.append(aug_img)
                    labels.append(int(label))
                
            images.append(img)
            labels.append(int(label))  # Use folder name as label

    return np.array(images), np.array(labels)


train_dir = './data/mask/train' #need to split your data into val (0 and 1) and train (0 and 1) sub-folders
val_dir = './data/mask/val'

# Load and preprocess images
train_images, train_labels = load_images_and_labels(train_dir)
val_images, val_labels = load_images_and_labels(val_dir)

# Create an instance of the model
model = efficientnet_binary()

#for small datasets make sure learning rate is not too high!
model.compile(optimizer=RMSprop(learning_rate=0.000005), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=40, batch_size=4, validation_data=(val_images, val_labels))

model.save('binary_efficientnet_b0_model.h5')
