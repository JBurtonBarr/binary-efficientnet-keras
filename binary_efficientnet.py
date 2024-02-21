import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D #Specific function calls to reduce package loading
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob



def efficientnet_binary(input_shape=(224, 224, 3), num_classes=1):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=output)

    return model


def create_dataset():
    # Image size and batch size, i.e for B0 we use 224 x 224
    img_size = (224, 224)
    batch_size = 32 #does not effect model params

    t_norm = ImageDataGenerator(rescale=1./255)#normalization is important
    v_norm = ImageDataGenerator(rescale=1./255)

    train_dataset = t_norm.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'  
    )

    val_dataset = v_norm.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )



train_dir = 'path/to/your/dataset/train' #need to split your data into val (0 and 1) and train (0 and 1) sub-folders
val_dir = 'path/to/your/dataset/validation'

# Create an instance of the model
model = efficientnet_binary()

model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

train_dataset, val_dataset = create_dataset(train_dir, val_dir)

# Train model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

model.save('binary_efficientnet_b0_model.h5')
