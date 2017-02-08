##Udacity P3 - DriveGG
###Towards a generalized Driving model based on data augmentation.

---

**Towards a more generalized Behavioral Cloning**

The goals of this project are the following:

* Using Udacity's simulator, collect and use the data to train a DNN.
* Augment the data, to consider multiple different scenarios.
* Output a desired steering angle, given an initial image input

Folder structure:
* backups: Contain other model weights and json files, that have similar results
* data: This is the same data set provided by Udacity 
* data2: This data set was created by me, using an Xbox controller.
* images: Images used in this README
* drive.py: The file called for driving. With little modifications from the original file
* model.py: Where training and data augmentation is performed.
* model.json: Outputs of the network
* model.h5: Outputs of the network
[//]: # (Image References)

[image1]: ./images/angle_count.png "Angle Count"
[image2]: ./images/angles_over_time.png "Angles over time"
[image3]: ./images/dataplot.png "Data Plot"
[image4]: ./images/brightness.png "Brightness change"
[image5]: ./images/translation.png "Shift image in x and y"
[image6]: ./images/crop_resize.png "Cropping and final resizing"

The following project goes complies with the [rubric](https://review.udacity.com/#!/rubrics/432/view) points
---

### Data generation and Augmentation 

For this assignment, two different datasets where used. One is the dataset released by Udacity and the other is data collected by myself using an Xbox controller. Both datasets have their differences, as it can be seen in the Angle count. This plot represent the angles present during the track for both datasets.

![alt text][image1]

As both images show, most of the angles are close to zero, which means that in most of the track the car is going straight. This is okay since the objective is to train the model to predict angles so that the car stays between the tracks, but also it's bad since we want the car to be trained with recoveries, meaning that the car can go back to the center after drifting a little.

For both images, the angles over time are expressed in the next graph:

![alt text][image2]

This graph also shows how many data points each set has. My own data has around 6000 images while Udacity's dataset has 8000. The datasets  themselves are not so different, but some angles can be noted, my dataset has values that shift a bit more, meaning that I make more turns and the values tend not to be that close to 0.

NOTE: Code for this graphs generation is located in: Visualizing and augmenting data.ipynb

Plotting some sample images in both datasets shows that the images are not very different:

![alt text][image3]

Having this data is not enough, in order to have a more generalized network we need to augment the datasets, by augmentation I mean that if we feed that dataset only to the network, it would tend to memorize the data, meaning that it will perform the first track ok, but would fail on the second one.

Keras, the framework using for this assignment, has very powerful image processing tools to augment one's data, further can be read [HERE](https://keras.io/preprocessing/image/). Also, a great example into data augmentation can be seen [HERE](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), the last one tackling the problem of differentiation of cats vs dogs. For this assignment, three different augmentation techniques are used: Shifting, brightness change and flipping.

#### Brightness change
In order to consider different lightning conditions and make the network more robust to dark environments or really bright ones, a brightness change augmentation is used. This function would receive an image, convert it to the hsv color space, and modify it's brightness by a random value.

The function that executes it goes as follow:

```python
def change_brightness(image):
    """Change brightness of an image for data augmentation. 
    :image: A RGB Image.
    Returns an RGB Image
    """
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    brightness = 0.20 + np.random.uniform()
    hsv_image[:,:,2] = hsv_image[:,:,2]* brightness
    return cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)
```

Testing this on some sample images give us:
![alt text][image4]

#### Translation in X and Y
Here, we shift the image in x and y, changing the angle only when a translation in x has been made.
These translations are at random, but some range of control is given, so that the image is not shifting in x or y more than a desired value. This kind of control values are used in the last function also, so that we can control if the image is totally dark or bright.

In this translation, opencv is used, with the `warpAffine` function, that receives a translation matrix M.

```python
def x_y_translation(image,angle):
    """Translate and image in X and Y plane for data augmentation. 
    :image: A RGB Image.
    :angle: The respective angle for that Image
    Returns a translated Image and the new_angle
    """
    x_translation = (X_RANGE * np.random.uniform()) - (X_RANGE * 0.5)
    y_translation = (Y_RANGE * np.random.uniform()) - (Y_RANGE * 0.5)
    # Translation Matrix
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    # M is the translation Matrix
    # M = np.float32([[1,0,X],[0,1,Y]])
    M = np.float32([[1,0,x_translation],[0,1,y_translation]])
    # Modify the angle for x, given input
    rows,cols,channels = image.shape
    translated_image = cv2.warpAffine(image,M,(cols,rows))
    new_angle = angle + ((x_translation/X_RANGE)*2)*ANGLE_RANGE
    return translated_image, new_angle
```
Some samples images changes look like this:
![alt text][image5]
It's good to know that the change in angle is only used when we do a translation in the X axis, because that is the only one to affect our angle. Translating in Y would not affect the angle at all.

#### Flipping the image through the vertical
One of the most simple tricks, but that could augment your data by 100% is only flipping the image with respect to the vertical, meaning that we will mirror the image. If we flip the image the angle is also changed, but the change is only the sign of the value of the angle.

```python
	# 1 in 3 chance to flip and image with respect to the vertical, and modify angle
    if np.random.randint(2) == 0: 
        image = np.fliplr(image)
        new_angle = -new_angle        
```
There is 1 in 3 chance to flip the image, the output is not shown since it's simple enough to imagine a mirroring change.

### Network architecture
The network is based on VGG16 [https://arxiv.org/pdf/1409.1556.pdf](https://arxiv.org/pdf/1409.1556.pdf). It's weights were loaded from there using Keras

| Layer                           |     Output Shape    |
|---------------------------------|---------------------|
| lambda_1 - Lambda               | (None, 64, 64, 3)   |
| block1_conv1 (Convolution2D)    | (None, 64, 64, 64)  |
| block1_conv2 (Convolution2D)    | (None, 64, 64, 64)  |
| block1_pool (MaxPooling2D)      | (None, 32, 32, 64)  |
| block2_conv1 (Convolution2D)    | (None, 32, 32, 128) |
| block2_conv2 (Convolution2D)    | (None, 32, 32, 128) |
| block2_pool (MaxPooling2D)      | (None, 16, 16, 128) |
| block3_conv1 (Convolution2D)    | (None, 16, 16, 256) |
| block3_conv2 (Convolution2D)    | (None, 16, 16, 256) |
| block3_conv3 (Convolution2D)    | (None, 16, 16, 256) |
| block3_pool (MaxPooling2D)      | (None, 8, 8, 256)   |
| block4_conv1 (Convolution2D)    | (None, 8, 8, 512)   |
| block4_conv2 (Convolution2D)    | (None, 8, 8, 512)   |
| block4_conv3 (Convolution2D)    | (None, 8, 8, 512)   |
| block4_pool (MaxPooling2D)      | (None, 4, 4, 512)   |
| block5_conv1 (Convolution2D)    | (None, 4, 4, 512)   |
| block5_conv2 (Convolution2D)    | (None, 4, 4, 512)   |
| block5_conv3 (Convolution2D)    | (None, 4, 4, 512)   |
| block5_pool (MaxPooling2D)      | (None, 2, 2, 512)   |
| Flatten 		                  | (None, 2048)        |
| fc1 (Dense)                     | (None, 1024)        |
| fc1_dropout (Dropout)           | (None, 1024)        |
| fc1 (Dense)                     | (None, 512)         |
| fc1_dropout (Dropout)           | (None, 512)         |
| fc2 (Dense)                     | (None, 256)         |
| fc2_dropout (Dropout)           | (None, 256)         |
| fc3 (Dense)                     | (None, 128)         |
| fc3_dropout (Dropout)           | (None, 128)         |
| fc4 (Dense)                     | (None, 64)          |
| fc4_dropout (Dropout)           | (None, 64)          |
| fc5 (Dense)                     | (None, 32)          |
| fc5_dropout (Dropout)           | (None, 32)          |
| output (Dense)                  | (None, 1)           |

### Training
The training is done through 10 iterations, penalizing small angles as the iterations go.
This penalization is done because as seen before, most the angles are close to 0, which would make the network really biased through that value, in order to avoid that, small angles close to 0 are penalized. This also has a drawback, because if we decrease the amount of angles close to 0, the network would zig zag a lot, a behavior that is not optimal and not recommended.

A python generator is used in order to load the data, it would be impossible to load all the images at once, memory would run out quickly. This generator is latter introduced into the training with the `fit_generator` function in Keras.

The best models are saved as the iterations go. If the model is already written and a better model is found, the function `save_best_model` would just rewrite it.

Our learning rate is of 1e-4. The training can be done with my_data or the udacity_data. There are some improvements from dataset to dataset, but also drawbacks too.

### Testing and generalization
The results are promising, having preloaded weights decreases the training time a lot. The network is generalized to the second track, with minor changes in the resolution.

### Discussion
The model tested here is a variant of vgg16, and there are many more promising models that would be interesting to implement, like [Nvidia's end to end driving model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) that would be interesting to use, maybe try and use a concatenated technique where 2 different networks are concatenated.


### Acknowledgments
The realization of this project would have not been possible without the help and inspiration of the following:
- The Slack channels, who provided lots of suggestions, and while some really useful ones, like adding random blobs of shadow to the images, or use a canny filter to pre process the image where not implemented, it's assumed that those would improve performance a lot.
- [Mohan Karthik](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.ubi8nrwor), [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.tyo1mko2e), [Matt Harvey](https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.ohvdnobar) and many other wonderful people for their great posts on tackling the assignment.
- Nvidia, for their support on getting a TX1, which will be used to deploy this network latter on.

