import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Add, ZeroPadding2D, Conv2D, Dropout, Activation
from keras.layers.core import Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.initializers import glorot_normal
from keras.optimizers import Adam

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

image_rows = 28; image_columns = 28
num_classes = 10

train_raw_data = pd.read_csv('train.csv')
test_raw_data = pd.read_csv('test.csv')

train_y = to_categorical(train_raw_data.label, num_classes)
train_X_raw = train_raw_data.values[:,1:].reshape(train_raw_data.shape[0], image_rows, image_columns, 1)
test_X_raw = test_raw_data.values.reshape(test_raw_data.shape[0], image_rows, image_columns, 1)

input_shape = train_X_raw.shape[1:]
train_X_input = Input(shape=input_shape)

train_X_shortcut_1 = train_X_input

train_X = Conv2D(8,kernel_size=(3,3),padding='same',strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv1')(train_X_input)
train_X = BatchNormalization(axis=-1, name='bn1')(train_X)
train_X = Dropout(0.6)(train_X)
train_X = Activation('relu')(train_X)

train_X = Conv2D(16,kernel_size=(3,3),padding='same',strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv2')(train_X)
train_X = BatchNormalization(axis=-1, name='bn2')(train_X)

train_X_shortcut_1 = Conv2D(16,kernel_size=(1,1),strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv1_sc')(train_X_shortcut_1)
train_X_shortcut_1 = BatchNormalization(axis=-1,name='bn1_sc')(train_X_shortcut_1)

train_X = Add()([train_X, train_X_shortcut_1])
train_X = Dropout(0.6)(train_X)
train_X = Activation('relu')(train_X)

train_X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(train_X)

train_X_shortcut_2 = train_X

train_X = Conv2D(32,kernel_size=(3,3),padding='same',strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv3')(train_X)
train_X = BatchNormalization(axis=-1, name='bn3')(train_X)
train_X = Dropout(0.6)(train_X)
train_X = Activation('relu')(train_X)

train_X = Conv2D(64,kernel_size=(3,3),padding='same',strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv4')(train_X)
train_X = BatchNormalization(axis=-1, name='bn4')(train_X)

train_X_shortcut_2 = Conv2D(64,kernel_size=(1,1),strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv2_sc')(train_X_shortcut_2)
train_X_shortcut_2 = BatchNormalization(axis=-1,name='bn2_sc')(train_X_shortcut_2)

train_X = Add()([train_X, train_X_shortcut_2])
train_X = Dropout(0.6)(train_X)
train_X = Activation('relu')(train_X)

train_X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(train_X)

train_X_shortcut_3 = train_X

train_X = Conv2D(128,kernel_size=(3,3),padding='valid',strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv5')(train_X)
train_X = BatchNormalization(axis=-1, name='bn5')(train_X)
train_X = Dropout(0.5)(train_X)
train_X = Activation('relu')(train_X)

train_X = Conv2D(256,kernel_size=(3,3),padding='valid',strides=(1,1),kernel_initializer=glorot_normal(seed=0),name='conv6')(train_X)
train_X = BatchNormalization(axis=-1, name='bn6')(train_X)

train_X_shortcut_3 = Conv2D(256,kernel_size=(1,1),strides=(3,3),kernel_initializer=glorot_normal(seed=0),name='conv3_sc')(train_X_shortcut_3)
train_X_shortcut_3 = BatchNormalization(axis=-1,name='bn3_sc')(train_X_shortcut_3)

train_X = Add()([train_X, train_X_shortcut_3])
train_X = Dropout(0.5)(train_X)
train_X = Activation('relu')(train_X)

train_X = Flatten()(train_X)
train_X = Dense(512,activation='relu')(train_X)
train_X = Dense(128,activation='relu')(train_X)
train_X = Dense(num_classes,activation='softmax')(train_X)

prediction_model = Model(inputs = train_X_input, outputs = train_X, name='Conv2DModel')

adam_lr_decay = Adam(lr=0.001, beta_1=0.75, beta_2=0.999, epsilon=1e-8, decay=5e-5)
prediction_model.compile(optimizer=adam_lr_decay,loss='categorical_crossentropy',metrics=['accuracy'])

prediction_model.fit(x=train_X_raw, y=train_y, batch_size=21, epochs=20, validation_split=0.0)

prediction_y = prediction_model.predict(test_X_raw)

prediction_result = pd.DataFrame({'ImageId': range(1,test_raw_data.shape[0]+1), 'Label': [np.argmax(prediction_y[i]) for i in range(test_raw_data.shape[0])]})
prediction_result.to_csv('prediction_submission.csv', index=False)
