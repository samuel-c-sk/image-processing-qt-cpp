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

### Linear convolution (5×5 smoothing filter)

This method applies a fixed 5×5 convolution kernel to the image.  
In practice, the chosen kernel behaves as a low-pass (smoothing) filter and reduces high-frequency noise and fine details.

#### Description

Linear convolution computes each output pixel as a weighted sum of its neighborhood using a predefined mask (kernel).  
Because the kernel weights are all non-negative and concentrated around the center, the operation performs spatial averaging and therefore produces a blurred / smoothed image.

#### Numerical principle

Let u(x, y) denote the normalized grayscale intensity at pixel location (x, y) in the interval [0, 1].  
Let w(i, j) be a 5×5 kernel with radius d = 2.

The convolution output is computed as:

    u_tilde(x, y) = sum_{i=-d}^{d} sum_{j=-d}^{d} w(i, j) · u(x + i, y + j)

To avoid boundary artifacts, the image is first extended using mirror padding (reflection at borders).  
After the convolution, values are clamped to [0, 1] to keep the output valid for grayscale visualization.

#### Implementation

- The input image is mirror-extended by d = 2 pixels on each side.
- The image is converted into a normalized double grid (values in [0, 1]).
- A fixed 5×5 kernel is applied at every pixel location.
- The result is converted back to an 8-bit grayscale QImage.

The current implementation uses a symmetric kernel with strong center weight, which effectively acts as a Gaussian-like smoothing filter.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
Upon activation, linear convolution is applied to the currently loaded image and the result is displayed.

#### Example output

`outputs/lena_lin_con_1.png`
`outputs/lena_lin_con_3.png`
`outputs/lena_lin_con_5.png`
`outputs/lena_lin_con_10.png`
`output/blurred_test_image_lin_con_40.png`

### Heat equation diffusion (explicit vs implicit scheme)

This method applies isotropic diffusion based on the 2D heat equation.  
In image processing, heat diffusion acts as a **smoothing (blurring) operator**: it reduces noise and high-frequency variations by averaging neighboring pixel values, while preserving large-scale image structures.

The user selects the time step size tau and the number of time steps T.  
Depending on the value of tau, the method uses either an explicit or an implicit time-stepping scheme.

#### Description

A grayscale image is treated as a discrete scalar field u(x, y) on a 2D grid.  
Diffusion is performed using a 5-point stencil (4-neighborhood), which corresponds to the discrete Laplacian operator.

As diffusion progresses in time, sharp intensity variations are smoothed out, resulting in an increasingly blurred image.

For stability reasons, the implementation selects:
- explicit scheme for tau <= 0.25
- implicit scheme for tau > 0.25

Mirror padding (reflection at borders) is used to handle boundary conditions and avoid edge artifacts.

#### Numerical principle

Let u^n(x, y) be the image at time step n and tau be the time step size.

The discrete Laplacian using the 4-neighborhood is:

    Δu(x, y) = u(x+1, y) + u(x-1, y) + u(x, y+1) + u(x, y-1) - 4u(x, y)

Explicit (forward Euler) update:

    u^{n+1}(x, y) = u^n(x, y) + tau · Δu^n(x, y)

which can be written as:

    u^{n+1}(x, y) = (1 - 4tau) u^n(x, y) + tau · (u^n_up + u^n_down + u^n_left + u^n_right)

Stability note (2D, 5-point stencil):
- the explicit scheme requires tau <= 0.25 for stable diffusion

Implicit (backward Euler) update:

    u^{n+1}(x, y) = u^n(x, y) + tau · Δu^{n+1}(x, y)

which leads to a linear system for u^{n+1} at each time step:

    (1 + 4tau) u^{n+1}(x, y) - tau · (u^{n+1}_up + u^{n+1}_down + u^{n+1}_left + u^{n+1}_right) = u^n(x, y)

#### Implementation

Explicit scheme:
- Uses mirror padding with radius d = 1.
- Computes each new pixel value from the 4-neighborhood using the explicit update formula.
- Repeats the update for T time steps.
- As time progresses, the image becomes progressively smoother (more blurred).

Implicit scheme:
- Uses an iterative solver (SOR-like relaxation) to compute u^{n+1} from the implicit linear system.
- Parameters used in the implementation:
  - maxIter = 100
  - omega (relaxation factor) = 1.25
  - tolerance tol = 1e-5
- At each time step, the method iterates until the residual norm drops below the tolerance (or max iterations are reached).

Diagnostics:
- The implementation prints the mean intensity before and after each time step.
- For the implicit scheme, it also logs the number of SOR iterations and the residual norm.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
The user sets:
- tau (time step size)
- T (number of time steps)

The program automatically selects explicit vs implicit scheme based on tau and displays the processed image.

#### Example output

`outputs/lena_HE_explicit_tau0.2_T_10.png`  
`outputs/lena_HE_implicit_tau0.8_T_10.png`
`outputs/blurred_test_image_HE_implicit_T20_tau0.8.png`
`outputs/blurred_test_image_HE_explicit_T20_tau0.2.png`

### Edge detection (gradient-based indicator)

This method computes a simple edge indicator based on local image gradients.  
Instead of producing a binary edge map, it generates a continuous edge-weight function in the interval (0, 1], where values close to 0 correspond to strong edges and values close to 1 correspond to flat (homogeneous) regions.

The parameter K controls the sensitivity of the detector: higher K makes the response drop faster near strong gradients.

#### Description

A grayscale image is treated as a discrete scalar field u(x, y).  
For each pixel, local gradient magnitudes are estimated using finite differences in four directions (east, north, west, south).  
These directional gradient magnitudes are averaged to obtain a single gradient strength measure.

The final output is computed using the nonlinear mapping:

    g = 1 / (1 + K · |grad(u)|^2)

This type of edge indicator is commonly used as a diffusivity function in edge-preserving diffusion models (e.g., Perona–Malik), where diffusion is reduced across edges.

#### Numerical principle

Let u(x, y) be the normalized grayscale intensity and let h be the grid spacing (here h = 1 pixel).

Directional gradients are approximated using finite differences.  
For example, for the "east" direction:

- x-component (forward difference):
    grad_east_x ≈ (u(x, y+1) - u(x, y)) / h

- y-component (central difference averaged across the east face):
    grad_east_y ≈ (u(x-1, y+1) - u(x+1, y+1) + u(x-1, y) - u(x+1, y)) / (4h)

The gradient magnitude for each direction is computed as:

    |grad_dir| = sqrt(grad_dir_x^2 + grad_dir_y^2)

The four directional magnitudes are averaged:

    grad_avg = (|grad_east| + |grad_north| + |grad_west| + |grad_south|) / 4

Finally, the edge indicator is:

    g(x, y) = 1 / (1 + K · grad_avg^2)

where K > 0 is a user-defined sensitivity parameter.

Boundary values are handled using mirror padding (reflection), so all finite differences are well-defined near the borders.

#### Implementation

- The input image is converted into a normalized double grid u in [0, 1].
- The image is mirror-extended by d = 1 pixel to support finite differences at the boundary.
- For each pixel, four directional gradient magnitudes are computed using finite differences.
- The final output is the edge indicator function g(x, y) in (0, 1].
- The result is converted back to an 8-bit grayscale image for visualization.

Note: this method produces an edge-weight map (a continuous field), not a thresholded binary edge image.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
The user provides the parameter K, and the resulting edge indicator map is displayed.

#### Example output

`outputs/lena_edges.png`

### Perona–Malik anisotropic diffusion (edge-preserving smoothing)

This method implements Perona–Malik anisotropic diffusion, an edge-preserving smoothing technique.  
Unlike the standard heat equation (isotropic diffusion), Perona–Malik reduces diffusion across strong gradients, which helps smooth homogeneous regions while keeping edges relatively sharp.

The practical effect is often perceived as "sharper edges" compared to standard blurring, because noise is reduced in flat areas but edges are not smoothed out as strongly.

#### Description

A grayscale image is treated as a discrete scalar field u(x, y).  
The method performs diffusion, but the diffusion strength depends on local gradient magnitude via an edge indicator function g.

In regions with small gradients (flat areas), g is close to 1 and diffusion behaves similarly to the heat equation (smoothing).  
Near strong gradients (edges), g becomes small and diffusion is reduced, which prevents edge blurring.

This produces denoising while preserving important structural boundaries.

#### Numerical principle

Let u^n(x, y) be the image at time step n and tau be the time step size.

Directional gradients are estimated using finite differences (east, north, west, south).  
For each direction, a conduction coefficient (edge indicator) is computed:

    g_dir(x, y) = 1 / (1 + K · |grad_dir(u)|^2)

where K controls sensitivity to edges (larger K -> stronger reduction of diffusion at edges).

The anisotropic diffusion update corresponds to a weighted discrete Laplacian, where neighbor contributions are scaled by g:

    (implicit form)  u^{n+1} = u^n + tau · div( g(|grad u|) · grad u )

In the discrete implementation, the update uses the 4-neighborhood with directional weights g_n, g_s, g_w, g_e:

    sum = g_n · u_up + g_s · u_down + g_w · u_left + g_e · u_right
    g_sum = g_n + g_s + g_w + g_e

and the implicit update is solved from:

    (1 + tau · g_sum) · u_new - tau · sum = u_old

Mirror padding is used to handle boundary conditions.

#### Implementation

The implementation proceeds in three main steps at each time step:

1) Optional pre-smoothing controlled by sigma
- If sigma is small, an explicit diffusion step is used.
- If sigma is larger, an implicit diffusion step is solved using SOR-like relaxation.
(This stabilizes gradient estimation before computing conduction coefficients.)

2) Gradient estimation and conduction coefficients
- The image is mirror-extended by d = 1 pixel.
- Directional gradient magnitudes are computed (east/north/west/south).
- Directional conduction coefficients g_e, g_n, g_w, g_s are computed using:
      g = 1 / (1 + K · |grad|^2)

3) Anisotropic diffusion update (implicit solve)
- The weighted diffusion step is solved iteratively using relaxation:
  - maxIter = 100
  - omega = 1.25
  - tol = 1e-5
- The residual norm is monitored to stop iterations early when convergence is reached.

Diagnostics:
- The implementation logs mean intensity per time step.
- It also logs the number of iterations and residual norm of the solver.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
The user sets:
- tau  (time step size for anisotropic diffusion)
- T    (number of time steps)
- sigma (pre-smoothing strength for gradient estimation)
- K    (edge sensitivity parameter for the conduction function)

The resulting edge-preserving smoothed image is displayed.

#### Example output

`outputs/blurred_test_image_PM_tau0.2_sigma0.5.png`

### Mean Curvature Flow (MCF)

This method implements mean curvature flow (MCF) for image smoothing.  
Compared to standard heat diffusion, MCF is a nonlinear geometric flow that tends to smooth structures while reducing curvature, often producing a more edge-aware / structure-preserving smoothing effect.

In practice, MCF can reduce noise and small oscillations while maintaining sharper transitions than pure isotropic blurring.

#### Description

A grayscale image is treated as a discrete scalar field u(x, y).  
Mean curvature flow can be interpreted as motion by curvature of level sets of the image intensity function.  
The update is nonlinear and depends on local gradient magnitude, which helps prevent excessive smoothing across strong edges.

To avoid numerical issues when gradients are small, a regularization parameter epsilon is used.

The method is implemented in an implicit-like form and solved iteratively using relaxation.

#### Numerical principle

Let u^n(x, y) be the image at time step n and tau be the time step size.

MCF can be written in continuous form as a geometric PDE (level-set interpretation):

    u_t = |grad u| · div( grad u / |grad u| )

In discrete form, the method uses directional gradient magnitudes computed by finite differences.
A regularized gradient magnitude is used:

    |grad_dir(u)|_eps = sqrt( (grad_dir_x)^2 + (grad_dir_y)^2 + epsilon^2 )

Directional weights are then defined as:

    g_dir = 1 / |grad_dir(u)|_eps

These weights reduce the influence of diffusion where gradients are strong, leading to a curvature-driven smoothing behavior.

An averaged gradient factor is computed:

    avg_grad_eps = (|grad_n|_eps + |grad_e|_eps + |grad_w|_eps + |grad_s|_eps) / 4

The update step is implemented in a weighted implicit form over the 4-neighborhood:

    sum   = g_n · u_up + g_s · u_down + g_w · u_left + g_e · u_right
    g_sum = g_n + g_s + g_w + g_e

and the iterative update solves approximately:

    (1 + tau · avg_grad_eps · g_sum) · u_new - tau · avg_grad_eps · sum = u_old

Mirror padding is used to handle boundary conditions.

#### Implementation

- The input image is converted into a normalized double grid u in [0, 1].
- The image is mirror-extended by d = 1 pixel.
- Directional gradients are computed (east/north/west/south) using finite differences.
- Regularized gradient magnitudes are computed using epsilon, and converted to weights g_e, g_n, g_w, g_s.
- The nonlinear update is solved iteratively using relaxation:
  - maxIter = 100
  - omega = 1.25
  - tol = 1e-5
- The residual norm is monitored to stop iterations when convergence is reached.

Diagnostics:
- The implementation logs the number of iterations and residual norm at each time step.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
The user sets:
- tau     (time step size)
- T       (number of time steps)
- epsilon (gradient regularization parameter)

The processed image is displayed after T steps.

#### Example output

`outputs/blurred_test_image_mcf_T20_tau0.4_eps_0.1.png`

### Gradient Mean Curvature Flow (GMCF)

This method implements Gradient Mean Curvature Flow (GMCF), a nonlinear, edge-aware smoothing technique that combines:
- curvature-driven regularization (similar to mean curvature flow), and
- gradient-based edge stopping (similar in spirit to Perona–Malik).

The practical goal is to smooth homogeneous regions while reducing smoothing near edges, preserving important structures better than isotropic blurring.

#### Description

A grayscale image is treated as a discrete scalar field u(x, y).  
The method computes directional gradients and uses them to build:
1) a curvature-related weighting term based on the regularized gradient magnitude, and
2) an edge-stopping term based on local gradient strength.

The resulting diffusion weights are direction-dependent and adapt locally:
- in flat regions, diffusion is stronger (smoothing/denoising),
- near strong gradients, diffusion is weaker (edge preservation).

An additional pre-smoothing step controlled by sigma is performed before computing the diffusion weights.

#### Numerical principle

Let u^n(x, y) be the image at time step n and tau be the time step size.

Regularized directional gradient magnitudes are computed:

    |grad_dir(u)|_eps = sqrt( (grad_dir_x)^2 + (grad_dir_y)^2 + epsilon^2 )

and an averaged factor:

    avg_grad_eps = (|grad_n|_eps + |grad_e|_eps + |grad_w|_eps + |grad_s|_eps) / 4

An edge-stopping factor is computed similarly to the edge indicator used in Perona–Malik:

    c_dir(x, y) = 1 / (1 + K · |grad_dir(u)|^2)

The final directional diffusion weights used by the method combine both effects:

    g_dir(x, y) = c_dir(x, y) · (1 / |grad_dir(u)|_eps)

This means diffusion is reduced:
- when gradients are strong (via c_dir),
- and when the regularized gradient magnitude is small/large (via 1 / |grad|_eps), producing curvature-type behavior.

Using the 4-neighborhood, the discrete update uses:

    sum   = g_n · u_up + g_s · u_down + g_w · u_left + g_e · u_right
    g_sum = g_n + g_s + g_w + g_e

The implicit weighted update is solved approximately from:

    (1 + tau · avg_grad_eps · g_sum) · u_new - tau · avg_grad_eps · sum = u_old

Mirror padding is used to handle boundary conditions.

#### Implementation

At each time step, the method proceeds as follows:

1) Regularized gradient magnitude (epsilon)
- Compute directional gradients and regularized magnitudes |grad_dir|_eps.
- Compute avg_grad_eps.

2) Pre-smoothing (sigma)
- If sigma is small, apply an explicit diffusion step.
- If sigma is larger, apply an implicit diffusion step solved by relaxation.
(This stabilizes the image before weight computation.)

3) Edge- and curvature-aware weights
- Recompute gradients after pre-smoothing.
- Compute c_dir = 1 / (1 + K · |grad_dir|^2).
- Compute final weights g_dir = c_dir · (1 / |grad_dir|_eps).

4) GMCF update (implicit solve)
- Solve the weighted implicit update using relaxation:
  - maxIter = 100
  - omega = 1.25
  - tol = 1e-5
- Stop when the residual norm falls below tol (or maxIter is reached).

Diagnostics:
- The implementation logs the number of solver iterations and residual norm per time step.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
The user sets:
- tau     (time step size)
- T       (number of time steps)
- sigma   (pre-smoothing strength)
- K       (edge sensitivity parameter)
- epsilon (gradient regularization parameter)

The processed image is displayed after T steps.

#### Example output

`outputs/blurred_test_image_gmcf_T20_tau0.4_eps0.1sig_0.1.png`

### Rouy–Tourin scheme (distance / level-set style evolution)

This method implements a Rouy–Tourin type iterative scheme that evolves a scalar field until convergence.  
In the context of image processing, the algorithm behaves like a distance / propagation process driven by local upwind differences, and it runs until no active pixels remain (instead of using a fixed number of time steps).

The output is a scalar field (stored as an image) which is normalized to [0, 1] for visualization.

#### Description

A grayscale image is treated as a discrete scalar field u(x, y).  
An auxiliary field `dist` is initialized using local neighborhood comparisons:
- pixels that differ from at least one of their 4-neighbors are marked as "active" (F = 0) and assigned distance 0
- pixels that match all 4 neighbors are initialized with a small positive value eps

Then, an iterative update is applied only to pixels that are still active (F = 1).  
During iterations, each pixel value is updated using an upwind-type expression based on local one-sided differences.

The algorithm stops when no pixels remain active (i.e., no updates are performed in an iteration).

#### Numerical principle

Let d(x, y) denote the evolving scalar field.  
At each iteration, for a pixel (x, y) the scheme uses one-sided differences and upwind selection:

    a = min(d(x-1,y) - d(x,y), 0)^2
    b = min(d(x+1,y) - d(x,y), 0)^2
    c = min(d(x,y-1) - d(x,y), 0)^2
    d = min(d(x,y+1) - d(x,y), 0)^2

and applies an update of the form:

    d_new(x, y) = d_old(x, y) + tau - (tau / h) · sqrt( max(a,b) + max(c,d) )

This is an explicit, monotone upwind-type update (often used in eikonal / front propagation schemes).
Pixels are marked as finished when the change becomes smaller than eps.

Boundary values are handled by copying border values outward (constant extension), ensuring that finite differences remain defined.

#### Implementation

- The input image is converted into a normalized double grid u in [0, 1].
- A padded distance field `dist` of size (H + 2d) × (W + 2d) is created (here d = 1).
- An activity mask F is initialized:
  - F[x][y] = 0 if the pixel differs from any 4-neighbor (edge / discontinuity indicator)
  - F[x][y] = 1 otherwise
- The algorithm iterates up to maxIter but stops early when no pixels remain active.
- After convergence, the inner part of `dist` is copied back into the output image grid.
- Because `dist` values are not naturally in [0, 1], they are normalized using min–max scaling before visualization.

#### Normalization helper

The function `normalizeDoubleMatrix` performs min–max normalization:

    normalized(x, y) = (value(x, y) - min) / (max - min)

If max == min, the range is set to 1 to avoid division by zero.

#### User interaction

The operation is triggered via a dedicated button in the graphical user interface.  
The user sets:
- tau (time step size)
- K (user parameter exposed in UI)

The iteration runs until convergence (no more active updates) or until the maximum iteration limit is reached.

Note: In the current implementation, K is passed as a parameter but not used inside the numerical update.

#### Example output

`outputs/blurred_test_image_RT.png`

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
