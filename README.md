# image-processing-qt-cpp

## 1. Project overview

This project focuses on the application of **numerical methods to image processing**, where a grayscale image is treated as a discrete signal or scalar field defined on a 2D grid.  
The main goal is to implement, test, and compare selected numerical and variational methods commonly used in image processing, such as diffusion-based smoothing and edge detection techniques.

The emphasis of the project is placed on the **correctness and behavior of individual numerical methods**, rather than on user interface design or visual appearance. The implementation serves primarily as a technical and educational exploration of these methods.

## 2. Input data

The input images used in this project are **grayscale images** stored in the repository under the `assets/` directory.  
They are used as test data for applying and evaluating individual image processing methods.

The images are intended for algorithmic experimentation and demonstration purposes only and do not represent production or application-specific datasets.


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
