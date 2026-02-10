# image-processing-qt-cpp

## 1. Project overview

This project focuses on the application of **numerical methods to image processing**, where a grayscale image is treated as a discrete signal or scalar field defined on a 2D grid.  
The main goal is to implement, test, and compare selected numerical and variational methods commonly used in image processing, such as diffusion-based smoothing and edge detection techniques.

The emphasis of the project is placed on the **correctness and behavior of individual numerical methods**, rather than on user interface design or visual appearance. The implementation serves primarily as a technical and educational exploration of these methods.

## 2. Input data

The input images used in this project are **grayscale images** stored in the repository under the `assets/` directory.  
They are used as test data for applying and evaluating individual image processing methods.

The images are intended for algorithmic experimentation and demonstration purposes only and do not represent production or application-specific datasets.

## 3. Numerical methods and image processing pipeline

### Image representation and data conversion

Before applying any numerical image processing methods, the input image is converted from a Qt image representation into a numerical form suitable for computation.

Images are treated as **2D scalar fields**, where pixel intensity values represent samples of a continuous function defined on a regular grid.

Two helper functions are used for this purpose:

#### Conversion from QImage to numerical grid

The function `imageToDouble` converts a grayscale `QImage` into a 2D array of double-precision values.

- Each pixel is converted to its grayscale intensity.
- Pixel values are normalized to the interval **[0, 1]**.
- The resulting data structure is a `std::vector<std::vector<double>>`, representing a discrete scalar field.

This representation allows direct application of numerical methods such as finite difference schemes, diffusion processes, and gradient-based operators.

#### Conversion from numerical grid back to image

The function `doubleToImage` performs the inverse operation, mapping a 2D array of normalized double values back to a grayscale image.

- Numerical values are rescaled to the interval **[0, 255]**.
- Values are clamped to avoid overflow or underflow.
- The output is stored as a `QImage` in 8-bit grayscale format.

This step enables visualization of numerical results and comparison between the original and processed images.

#### Notes
These conversion routines do not modify the image content by themselves.  
They provide a consistent numerical interface between the image representation and the implemented image processing methods.

### Boundary handling and image extension

Several numerical image processing methods require access to pixel values outside the original image domain (e.g. diffusion schemes, convolution-based operators).  
To handle boundary conditions in a consistent way, mirror-based image extension is used.

The following helper functions implement mirror padding and are used internally by multiple processing methods.

#### Mirror padding

Mirror padding extends the image domain by reflecting pixel values across the image boundaries.  
This approach avoids artificial discontinuities at the borders and provides a smooth continuation of the image intensity field.

The padding width is controlled by the parameter `d`, which specifies the number of pixels added on each side.

#### Numerical principle

Let u(x, y) denote the image intensity defined on a discrete grid.

For pixels outside the original domain, values are obtained by symmetric reflection with respect to the image boundary.  
This corresponds to enforcing Neumann-type boundary conditions, where the normal derivative at the boundary is approximately zero.

Mirror padding is commonly used in numerical schemes to prevent boundary artifacts and ensure stable behavior of local operators.

#### Implementation

Two overloaded versions of the mirror padding function are implemented:

- A version operating on `QImage`, used for visualization and UI interaction.
- A version operating on a 2D numerical grid (`std::vector<std::vector<double>>`), used by numerical algorithms.

The implementation performs the following steps:
- The original image is copied into the center of an extended image domain.
- Top and bottom boundaries are filled by reflecting pixel values vertically.
- Left and right boundaries are filled by reflecting pixel values horizontally.

The function does not modify the original image content; it only extends the computational domain to support subsequent numerical operations.

### Full-Strength Histogram Stretching (FSHS)

Full-strength histogram stretching is a basic intensity normalization technique used to improve image contrast by expanding the range of grayscale values.

#### Description

In this method, the grayscale image is rescaled such that the minimum intensity value present in the image is mapped to 0, and the maximum intensity value is mapped to 1.  
All intermediate pixel values are linearly transformed accordingly.

This operation enhances contrast by utilizing the full available intensity range without altering the relative ordering of pixel values.

#### Numerical principle

Let u(x, y) denote the normalized grayscale intensity at pixel location (x, y).

The transformed image is computed as:

    u_tilde(x, y) = (u(x, y) - min(u)) / (max(u) - min(u))

where:
- min(u) is the minimum intensity value in the image,
- max(u) is the maximum intensity value in the image.

This corresponds to an affine transformation of the intensity values.

#### Implementation

The image is first converted into a numerical representation using a normalized double-precision grid.  
The global minimum and maximum intensity values are determined by scanning the entire image.

Each pixel value is then rescaled using the formula above and converted back to a grayscale image for visualization.

The method does not introduce any smoothing or edge enhancement; it only redistributes intensity values.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
Upon activation, the current image is processed using the FSHS method and immediately displayed.

#### Example output

`outputs/lena_fshs.png`

### Histogram Equalization (EH)

Histogram equalization is a global contrast enhancement technique that redistributes grayscale intensity values based on their cumulative distribution in the image.

Unlike simple linear rescaling, this method uses the statistical distribution of intensities to achieve a more uniform histogram.

#### Description

In this method, the grayscale image is transformed using the cumulative distribution function (CDF) of its intensity histogram.  
The goal is to spread frequently occurring intensity values over a wider range, improving visibility of details in low-contrast regions.

The relative ordering of pixel intensities is preserved, but the spacing between intensity levels is nonlinearly modified.

#### Numerical principle

Let u(x, y) denote the grayscale intensity at pixel location (x, y), with values in {0, …, 255}.

First, the histogram of intensity values is computed and used to construct the cumulative distribution function (CDF):

    cdf(i) = (1 / (W · H)) · sum_{j=0}^{i} hist(j)

where:
- W · H is the total number of pixels,
- hist(i) is the number of pixels with intensity i.

Each pixel value is then remapped using the normalized CDF:

    u_tilde(x, y) = 255 · cdf(u(x, y))

This nonlinear transformation redistributes intensity values so that the resulting histogram is approximately uniform.

#### Implementation

A histogram with 256 bins is constructed from the input image.  
The cumulative distribution function is computed and normalized by the total number of pixels.

Each pixel intensity is then replaced by its CDF-based value scaled to the full grayscale range [0, 255].

The method operates on grayscale images and does not introduce any spatial smoothing.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
Upon activation, histogram equalization is applied to the currently loaded image and the result is displayed.

#### Example output

`outputs/lena_EH.png`

## 5. Code structure

The project is structured to separate **numerical image processing logic** from **visualization and user interface components**.

- Core image processing algorithms are implemented in dedicated source files (e.g. `ImageProcessing.*`).
- Visualization and interaction with images are handled separately through Qt-based viewer and widget classes.
- The main application logic initializes the interface and connects user actions to the underlying numerical methods.

This separation allows the numerical methods to be inspected and evaluated independently of the graphical user interface.

## 6. Notes on scope and focus

The primary objective of this project was to **verify the functionality and numerical behavior of individual image processing methods**.  
As a result:

- Code style, optimization, and UI aesthetics were not the main focus.
- The graphical interface serves only as a tool for applying methods and visualizing results.
- The implementation prioritizes clarity of numerical procedures over production-level software design.

This scope was chosen deliberately to focus on understanding and validating the underlying numerical techniques.

## 7. User interface (reference only)

A simple graphical user interface implemented using **Qt** is included in the project to allow basic interaction with images and visualization of results.

The UI was not designed with usability or visual polish as a primary goal.
Screenshot of the user interface can be found in the `screenshots/` directory. 
Relevant UI definitions and source files can be found directly in the repository (e.g. `.ui`, viewer, and widget source files).

## 8. Disclaimer and context

This project was developed in an **academic context** as part of coursework focused on numerical methods and image processing.  
It is intended as a demonstration of algorithmic understanding and implementation rather than as a production-ready image processing application.
