# Digital Image Processing

Digital Image Processing (DIP) is the use of computer algorithms to perform operations on images for tasks like enhancement, analysis, and transformation.


## OpenCV
OpenCV (Open Source Computer Vision Library) is an open-source library focused on computer vision and image processing tasks. It provides tools and functions for real-time image processing.

## Numpy

NumPy (Numerical Python) is a powerful Python library for numerical computing. It provides support for working with large, multi-dimensional arrays and matrices, along with a vast collection of mathematical functions to operate on these arrays efficiently.

## Code Exlaination:

```python
# import library
import numpy as np
import cv2 as cv

img = cv.imred("imgPath") # to read image in the form of arrays

# convert color
img = cv.cvtColor(image, colorMode)
```

```python
px = img[100,100]
print( px )

# accessing only blue pixel
blue = img[100,100,2]
print( blue )

img[100,100] = [255,255,255]
print( img[100,100] )

```

- `img[100, 100]` accesses the pixel located at the (100, 100) position in the image.
- `img[100, 100, 2]` accesses only the blue channel (index 2) of the pixel at (100, 100).
- `[255, 255, 255]` represents a white color in the RGB color model, where each channel (Red, Green, Blue) is at its maximum intensity (255).
- After this line, the pixel at (100, 100) will be white.
- `img.shape` is an attribute provide tuple of (height, width, channels).
- `img.size` is an attribute
- `img.dtype` for img datatype
- cv.split(img) splits the color image into its individual color channels (typically Blue, Green, and Red in OpenCV).
- cv.merge() combines separate single-channel images into one multi-channel image.
- `b = img[:, :, 0]` is used to access the blue channel of the img image directly.


## cv.copyMakeBorder()
```python
bordered_img = cv.copyMakeBorder(src, top, bottom, left, right, borderType, value)

# cv.BORDER_CONSTANT: Adds a constant-colored border. The color is defined by value.
# cv.BORDER_REFLECT: Reflects the border pixels (e.g., fedcba|abcdefgh|hgfedcba).
# cv.BORDER_REFLECT_101 or cv.BORDER_DEFAULT: Similar to BORDER_REFLECT but without repeating the last border element.
# cv.BORDER_REPLICATE: Replicates the last pixel of the image in each direction (e.g., aaaaaa|abcdefgh|hhhhhhh).
# cv.BORDER_WRAP: Wraps around to the opposite edge of the image.

```

## cv.addWeighted
```python

dst = cv.addWeighted(src1, alpha, src2, beta, gamma)

dst = cv.addWeighted(img1,0.7,img2,0.3,0)

cv.imshow(dst) # to show image
```

## Image Operation
Addition (cv.add())

Adds two images or an image and a scalar value.

### Addition (cv.add())

- Adds two images or an image and a scalar value.
- Pixel values are added together. Overflow is handled automatically by OpenCV.
- If the sum exceeds the maximum value (255 for an 8-bit image), the value will be clipped to 255.

result = cv.add(img1, img2)

### Subtraction (cv.subtract())
- Subtracts one image from another or a scalar value from an image.
- Negative values are clipped to 0 in the result.

result = cv.subtract(img1, img2)

## Multiplication (cv.multiply())

- Multiplies two images or an image by a scalar.
- Pixel values are multiplied together. If the result exceeds the maximum value (255 for 8-bit images), it is clipped to 255.
- result = cv.multiply(img1, img2)
### Division (cv.divide())

- Divides one image by another or an image by a scalar.
- Division by zero is avoided by OpenCV, and the result is clipped to the valid range.
- result = cv.divide(img1, img2)
- Scalar Operations

- You can also perform arithmetic operations between an image and a scalar value, such as adding a constant to all pixel values or multiplying all pixels by a constant.
- result = cv.add(img, 50)  # Add 50 to each pixel of the image
- result = cv.multiply(img, 2)  # Multiply each pixel by 2

## Bitwise

Bitwise AND (cv.bitwise_and())

Performs a bitwise AND operation on corresponding pixels of two images.
Only the bits that are set (1) in both images remain set.
result = cv.bitwise_and(img1, img2)
Bitwise OR (cv.bitwise_or())

Performs a bitwise OR operation on corresponding pixels of two images.
Bits that are set (1) in either of the images will remain set.
result = cv.bitwise_or(img1, img2)
Bitwise XOR (cv.bitwise_xor())

Performs a bitwise XOR (exclusive OR) operation on corresponding pixels of two images.
Bits that are different (one image has 1 and the other has 0) will be set to 1.
result = cv.bitwise_xor(img1, img2)
Bitwise NOT (cv.bitwise_not())

Inverts the bits of the image.
This operation flips all bits in the image (i.e., 1 becomes 0, and 0 becomes 1).
result = cv.bitwise_not(img)


## Geometric Transformation Of Images

Geometric transformations in image processing refer to operations that modify the spatial arrangement of the pixels in an image. These transformations can change the size, orientation, position, or shape of an image. In OpenCV, geometric transformations are frequently used for tasks such as image resizing, rotating, translating, and applying perspective changes.

#### Summary of Geometric Transformation Techniques
Translation: Shifts the image in space.
Scaling: Resizes the image based on a scaling factor.
Rotation: Rotates the image around a specific center point.
Affine Transformation: Combines translation, scaling, and rotation with shearing.
Perspective Transformation: Changes the viewpoint of the image, simulating a 3D effect.
These transformations are essential for various tasks like image alignment, data augmentation, and visual effects in computer vision applications.

## 2D Convolution
2D Convolution is a fundamental operation in image processing and computer vision. It involves applying a kernel (also known as a filter) to an image in order to extract important features such as edges, corners, and textures, or to perform tasks like blurring, sharpening, and edge detection.

## How 2D Convolution Works
- Input Image: The image to which the convolution is applied (a matrix of pixel values).
- Kernel (Filter): A small matrix (usually odd-sized, e.g., 3x3, 5x5) that contains weights used for combining neighboring pixel values.
- Sliding Window: The kernel is moved (slid) over the image, pixel by pixel. At each position, the kernel "convolves" with the image, performing a mathematical operation (typically dot product).
- Output Image: The result of the convolution, which is a new image where each pixel represents the transformed value based on the kernel and its neighbors.

![image](https://github.com/user-attachments/assets/4be491a4-428f-466a-a224-b31bbc4ccdbe)


**Applying Convolution in OpenCV**
OpenCV provides functions to apply 2D convolution using different filters.

`cv2.filter2D():`

This function applies a custom kernel to an image using 2D convolution.
Syntax:

`dst = cv2.filter2D(src, ddepth, kernel)`

- src: The input image.
- ddepth: The desired depth of the destination image (usually -1 to use the same depth as the input).
- kernel: The convolution kernel (filter).

### Applications of 2D Convolution
- Edge Detection: Detect edges using filters like the Sobel filter.
- Blurring/Filtering: Smooth an image using a box blur or Gaussian blur.
- Sharpening: Enhance edges by using sharpening filters.
- Feature Extraction: Extract important features (e.g., textures) for tasks like object recognition or segmentation.

blur = cv.GaussianBlur(img,(5,5),0) # Guassian Blur

median = cv.medianBlur(img,5) # mediaum Blurring

blur = cv.bilateralFilter(img,9,75,75) # Bilateral Filtering

**Morphological transformations** are a set of operations used in image processing to process images based on their shape or structure. These operations are particularly useful for binary or grayscale images and are often used for tasks such as noise removal, hole filling, edge detection, and feature extraction. They work by probing the image with a structuring element (a small matrix, often square or circular), which is applied to each pixel and its neighbors.

### Key Morphological Operations in OpenCV:

1. **Erosion**
2. **Dilation**
3. **Opening**
4. **Closing**
5. **Morphological Gradient**
6. **Top Hat**
7. **Black Hat**

### 1. **Erosion**
- **Erosion** shrinks the white regions in a binary image. The kernel slides over the image, and at each position, the output pixel is set to the minimum value of the pixels in the kernel's neighborhood.
- It is often used to remove small noise from an image or reduce the size of the foreground object.

**Effect**: Erosion erodes away the boundaries of the object in the image.

**OpenCV function**: `cv2.erode()`

**Example**:
```python
import cv2
import numpy as np

# Load an image (binary image, for example, after thresholding)
img = cv2.imread('binary_image.jpg', 0)

# Define a 3x3 kernel
kernel = np.ones((3, 3), np.uint8)

# Apply erosion
eroded_img = cv2.erode(img, kernel, iterations=1)

# Show the result
cv2.imshow('Eroded Image', eroded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2. **Dilation**
- **Dilation** is the opposite of erosion. It increases the white regions in a binary image. The output pixel is set to the maximum value of the pixels in the kernel's neighborhood.
- It is useful for filling small holes in objects or connecting disjoint parts of an object.

**Effect**: Dilation expands the boundaries of the foreground object.

**OpenCV function**: `cv2.dilate()`

**Example**:
```python
# Apply dilation
dilated_img = cv2.dilate(img, kernel, iterations=1)

# Show the result
cv2.imshow('Dilated Image', dilated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3. **Opening**
- **Opening** is an operation that involves **erosion** followed by **dilation**. It is useful for removing small noise from the image while preserving the shape and size of larger objects.
- It can be used to eliminate small objects or holes.

**Effect**: Removes small objects from the foreground (shrinking objects).

**OpenCV function**: `cv2.morphologyEx()`

**Example**:
```python
# Apply opening (erosion followed by dilation)
opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Show the result
cv2.imshow('Opened Image', opened_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 4. **Closing**
- **Closing** is an operation that involves **dilation** followed by **erosion**. It is typically used to close small holes or gaps in objects in the foreground while preserving the overall shape.
- It can help fill small holes inside objects or join nearby objects.

**Effect**: Fills small holes and gaps in the foreground.

**OpenCV function**: `cv2.morphologyEx()`

**Example**:
```python
# Apply closing (dilation followed by erosion)
closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Show the result
cv2.imshow('Closed Image', closed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 5. **Morphological Gradient**
- The **morphological gradient** is the difference between the dilation and erosion of an image. It highlights the boundaries of objects in the image.
- It is often used to detect edges.

**Effect**: Highlights edges by computing the difference between dilation and erosion.

**OpenCV function**: `cv2.morphologyEx()`

**Example**:
```python
# Apply morphological gradient
gradient_img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Show the result
cv2.imshow('Morphological Gradient', gradient_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 6. **Top Hat**
- The **top hat** operation is the difference between the input image and the opening of the image. It is useful for extracting small bright objects from a darker background.
- It highlights small features that are lighter than their surroundings.

**Effect**: Extracts small bright regions from the image.

**OpenCV function**: `cv2.morphologyEx()`

**Example**:
```python
# Apply top hat operation
tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Show the result
cv2.imshow('Top Hat Image', tophat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 7. **Black Hat**
- The **black hat** operation is the difference between the closing of the image and the input image. It is useful for extracting small dark objects from a lighter background.
- It highlights small features that are darker than their surroundings.

**Effect**: Extracts small dark regions from the image.

**OpenCV function**: `cv2.morphologyEx()`

**Example**:
```python
# Apply black hat operation
blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# Show the result
cv2.imshow('Black Hat Image', blackhat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Structuring Elements
- The **structuring element** is the kernel or mask used in morphological operations. It is usually a small binary image (matrix), such as a square, circle, or ellipse, that defines the neighborhood around each pixel.
- The most commonly used structuring elements are:
  - **Rectangular**: `np.ones((3, 3), np.uint8)` (3x3 matrix of ones)
  - **Elliptical**: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))`
  - **Cross**: `cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))`

### Summary of Key Morphological Operations

1. **Erosion**: Reduces white regions; removes small noise.
2. **Dilation**: Expands white regions; fills small holes.
3. **Opening**: Erosion followed by dilation; removes small objects and noise.
4. **Closing**: Dilation followed by erosion; fills small holes and gaps.
5. **Morphological Gradient**: Difference between dilation and erosion; highlights edges.
6. **Top Hat**: Difference between the original image and the opening; highlights small bright regions.
7. **Black Hat**: Difference between the closing and the original image; highlights small dark regions.

These operations are commonly used in tasks like object detection, noise removal, boundary detection, and feature extraction in image analysis.


