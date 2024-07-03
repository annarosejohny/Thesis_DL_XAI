import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras import backend as K

# Define train and test images directory
TRAIN_DIR = 'train_custom/'
TEST_DIR = 'test_custom/'


#Define global variables
H = 100
W = 100
epochs = 25
batch_size = 32
SEED = 42
height, width = 100,100
training_batch_size=32
images = []
labels = []
image_filenames = []
image_filenames_test = []
label_tests = []
image_test = []


def calculate_iou(pred_mask, gt_mask, true_pos_only=False):
  #Calculate IoU score between two segmentation masks.

  intersection = np.logical_and(pred_mask, gt_mask)
  union = np.logical_or(pred_mask, gt_mask)

  if true_pos_only:
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
      iou_score = np.nan
    else:
      iou_score = np.sum(intersection) / (np.sum(union))
  else:
    if np.sum(union) == 0:
      # ! union has no overlapping, so return 0 intead of nan ??
      # ! at extreme threshold, this happens if we put in all black images (so 0 or 0 = 0)
      iou_score = 0 # np.nan
    else:
      iou_score = np.sum(intersection) / (np.sum(union))

if __name__ == "__main__":
  
  # training data
  class_idx_to_label = {"22q11DS":0, "BWS":1, "CdLS":2, "Down":3, "KS":4,"NS":5, "PWS":6, "RSTS1":7, "Unaffected":8, "WHS":9, "WS":10}
  train_df = pd.DataFrame({"image_id": image_filenames, "label": labels})
  train_df['class_idx_to_label'] = train_df['label'].map(class_idx_to_label)
  train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
  train_generator = train_datagen.flow_from_dataframe(train_df,TRAIN_DIR,x_col="image_id", y_col="label",
                                                    target_size=(W,H), class_mode="categorical",
                                                   batch_size=batch_size, shuffle=True, seed=SEED)

  train_set = tf.keras.preprocessing.image_dataset_from_directory('/train_custom',seed=123,image_size=(height, width),batch_size=training_batch_size)

  validation_set = tf.keras.preprocessing.image_dataset_from_directory('/test_custom',seed=123,image_size=(height, width),batch_size=training_batch_size)

  # testing data
  test_df = pd.DataFrame({"image_id": image_filenames_test, "label": label_tests})
  test_df['class_idx_to_label'] = test_df['label'].map(class_idx_to_label)
  test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
  test_generator = train_datagen.flow_from_dataframe(test_df,TEST_DIR,x_col="image_id", y_col="label",
                                                    target_size=(W,H), class_mode="categorical",
                                                   batch_size=batch_size, shuffle=True, seed=SEED)
  
  
  # defining the model
  imported_model= tf.keras.applications.ResNet50(include_top=False,
  input_shape=(100,100,3),pooling='avg',classes=11,weights=None)

  for layer in imported_model.layers:
    layer.trainable=False
  
  model = Sequential()
  model.add(imported_model)
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(5, activation='softmax'))

  # train the model using train and validation sets for 10 epochs. Few number of epochs are taken because of the smaller datasize
  model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  history = model.fit(train_set,validation_data=validation_set,epochs=10)
  
  # load images from test dataset and obtain predictions from the test set
  image=cv2.imread('/test_custom')
  image_resized= cv2.resize(image, (height, width))
  image=np.expand_dims(image_resized,axis=0)

  model_pred = model.predict(image)
  class_names = train_set.class_names
  predicted_class=class_names[np.argmax(model_pred)]
  print("The predicted category is", predicted_class)

  class_probabilities = model_pred[0]  # Assuming predictions is a list with a single prediction (batch size of 1)

  # if you want to know the class labels as well
  class_labels = ["22q11DS", "BWS", "CdLS", "Down", "KS", "NS", "PWS", "RSTS1", "Unaffected", "WHS", "WS"]

  # zip class labels with probabilities
  class_probabilities_with_labels = list(zip(class_labels, class_probabilities))

  # display class probabilities
  for class_label, probability in class_probabilities_with_labels:
    print(f"Class: {class_label}, Probability: {probability}")

  # generating heatmaps using Gradcam approach. Heatmaps are generated based on the decisions from the last convolutional layer
  img_bgr = cv2.imread('/CdLS/CdLSSlide124_crop_square.jpg') 
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

  img_resized = cv2.resize(img_rgb, (100,100), interpolation = cv2.INTER_CUBIC)

  img_tensor = np.expand_dims(img_resized, axis=0)

  img_tensor = tf.keras.applications.resnet50.preprocess_input(img_tensor)
  
  heatmap_model = Model(model.input,outputs = [model.layers[-40].output, model.output])

  with tf.GradientTape() as gtape:
    conv_output, predictions = heatmap_model(img_tensor)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = gtape.gradient(loss, conv_output)
    pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2)) 
  heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)

  heatmap = np.maximum(heatmap, 0)
  max_heat = np.max(heatmap)
  if max_heat == 0:
    max_heat = 1e-10
  heatmap /= max_heat
  print(heatmap)

  squeezed_hm = np.squeeze(heatmap)

  normalized_hm = cv2.resize(squeezed_hm, (img_bgr.shape[1], img_bgr.shape[0]))

  normalized_hm = (255 * normalized_hm).astype(np.uint8)
  normalized_hm = cv2.applyColorMap(normalized_hm, cv2.COLORMAP_JET)

  superimposed_img = cv2.addWeighted(normalized_hm, 0.5, img_bgr, 0.5, 0)
  plt.imshow(cv2.cvtColor(normalized_hm, cv2.COLOR_BGR2RGB))
  plt.axis('off')
  plt.show()  

  # IOU score calculation between two heatmaps
  image_clinician = '/non_clinician.png'
  image_non_clinician = '/nonclinician_cdLS.png'

  image_clinician = cv2.imread(image_clinician)

  height, width, channels = image_clinician.shape
  print(height, width, channels)
  cv2.imshow(cv2.cvtColor(image_clinician, cv2.COLOR_BGR2GRAY))
  image_gradcam = cv2.imread('/BWS_Custom.png')
  height, width, channels = image_gradcam.shape
  print(height, width, channels)
  cv2.imshow(cv2.cvtColor(image_gradcam, cv2.COLOR_BGR2GRAY))

  iou_score = calculate_iou(image_clinician,image_gradcam,true_pos_only=False)
  print (iou_score)




