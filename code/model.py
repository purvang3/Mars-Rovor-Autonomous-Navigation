import csv
import cv2
import numpy as np
import sklearn
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    return warped

def process_img (image):
    dst_size = 12
    src = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]])
    dst = np.float32([[320 / 2 - dst_size, 160],
                      [320 / 2 + dst_size, 160],
                      [320 / 2 + dst_size, 160 - 2 * dst_size],
                      [320 / 2 - dst_size, 160 - 2 * dst_size],
                      ])

    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    path_threshed = color_thresh(hsv_img, rgb_thresh=(0, 0, 90))
    sample_threshed = color_thresh(hsv_img, rgb_thresh=(0, 120, 120))

    path_warped = perspect_transform(path_threshed, src, dst)
    #sample_warped = perspect_transform(sample_threshed, src, dst)

    drive_img = np.dstack((path_threshed, sample_threshed, path_warped)).astype(np.uint8)

    return drive_img


def translate_image(img, line):
    # Translation of image data to create more training data
    # img : 3D image data
    # y_value : float label data for the image
    # distance : max distance of transformed images
    # step : steps between transformed images
    # y_value_gain : the gain of the label data over the transform distance
    # return : list of new images and corresponding label data

    rows, cols, _ = img.shape
    img_list = []
    data_list = []

    # add original un touched image
    img_list.append(img)
    data_list.append((float(line[1]), float(line[2]), float(line[3])))

    # add step to include the distance in the image transform
    for offset in range(20, 100, 20):
        M_right = np.float32([[1, 0, offset], [0, 1, 0]])  # move image right
        M_left = np.float32([[1, 0, -offset], [0, 1, 0]])  # move image left

        # shift the image to the right and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_right, (cols, rows)))
        # shift the image to the left and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_left, (cols, rows)))

        data_list.append((float(line[1]), float(line[2]), float(line[3])))  # add positive steer angle for a right shift
        data_list.append((float(line[1])*-1, float(line[2]), float(line[3])))  # add negative steer angle for a left shift

    return img_list, data_list

drive_data = []
control_data = []
# Create normalised images for the training data
print('Creating the data')
with open('./data/robot_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        angle = float(line[1])

        # only keep data with a steer angle
        if angle >= 0.01 or angle <= -0.01:
            name = './data/IMG/' + line[0].split('/')[-1]
            image = cv2.imread(name)

            img_list, angle_list = translate_image(image, line)
            drive_data.extend(img_list)
            control_data.extend(angle_list)

            line[1] = float(line[1]) * -1
            img_list, angle_list = translate_image(np.fliplr(image), line)
            drive_data.extend(img_list)
            control_data.extend(angle_list)

print('data count: ', np.shape(drive_data))
train_data, val_data, train_labels, val_labels = train_test_split(drive_data,
                                                                  control_data,
                                                                  test_size=0.1)
print('Finished collecting Data')


def generator(x_data, y_data, batch_size=16):
    # generates the data for run time memory efficiency
    # x_data : features input data
    # y_data : label data for the x_data
    # batch_size : size of each batch of data to be trained
    # return : batch sized, shuffled data for training
    num_samples = len(x_data)
    print(num_samples)
    while 1:  # Loop forever so the generator never terminates
        shuf_img, shuf_angle = shuffle(x_data, y_data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = shuf_img[offset:offset + batch_size]

            batch_images = []
            for data in batch_samples:
                img = process_img(data)
                batch_images.append(img)

            # trim image to only see section with road
            X_train = np.array(batch_images)
            y_train = np.array(shuf_angle[offset:offset + batch_size])

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_data, train_labels, batch_size=32)
validation_generator = generator(val_data, val_labels, batch_size=32)

print('Starting the model')
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# create the model
x_input = Input(shape=(160, 320, 3))
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x_input)
x = Convolution2D(24, 5, 5,
                  subsample=(2, 2),
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(36, 5, 5,
                  subsample=(2, 2),
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(48, 5, 5,
                  subsample=(2, 2),
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(64, 3, 3,
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(64, 3, 3,
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)
x = Dropout(0.4)(x)
x = Convolution2D(64, 3, 3,
                  activation='elu',
                  border_mode='valid',
                  dim_ordering='tf',
                  W_constraint=maxnorm(3))(x)
x = BatchNormalization(epsilon=0.001,
                       mode=0,
                       axis=2,
                       momentum=0.99)(x)

x = (Flatten())(x)
print(np.shape(x))
x = (Dense(100))(x)
x = (ELU(alpha=1.0))(x)
x = (Dropout(0.4))(x)
x = (Dense(50))(x)
x = (ELU(alpha=1.0))(x)
out = (Dense(3))(x)

model = Model(input=x_input, output=out)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_data),
                    validation_data=validation_generator,
                    nb_val_samples=len(val_data),
                    nb_epoch=4)

print('Saving model')
model.save('model_hard.h5')

print('Finished')
cv2.waitKey(0)
cv2.destroyAllWindows()
gc.collect()