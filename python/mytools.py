import h5py
import numpy as np
import SimpleITK as sitk

def read_h5_to_double(file_path, dataset_name=None):
    with h5py.File(file_path, 'r') as f:
        # If dataset_name is not provided, try to get the first dataset
        if dataset_name is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"HDF5 file {file_path} is empty, no datasets available")
            dataset_name = keys[0]
        
        # Read the dataset
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        
        data = f[dataset_name][()]
        
        # Ensure data is 3D
        if len(data.shape) != 3:
            raise ValueError(f"Data is not 3D. Actual dimensions: {data.shape}")
        
        # Convert data to float type
        return data.astype(np.float64)
    
def read_h5_to_float(file_path, dataset_name=None):
    with h5py.File(file_path, 'r') as f:
        if dataset_name is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"HDF5 file {file_path} is empty, no datasets available")
            dataset_name = keys[0]
        
        data = f[dataset_name][()]
        return data.astype(np.float32)
    
def save_h5_from_float(file_path, dataset_name, data):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset_name, data=data, dtype=np.float32)

def numpy_to_sitk_image(array):
    """Convert numpy array to SimpleITK image"""
    # SimpleITK expects (width, height) order, numpy uses (height, width)
    if len(array.shape) == 2:
        array = array.T
    
    image = sitk.GetImageFromArray(array.astype(np.float32))
    return image

def sitk_image_to_numpy(image):
    """Convert SimpleITK image to numpy array"""
    array = sitk.GetArrayFromImage(image)
    # Convert back to (height, width) for 2D
    if len(array.shape) == 2:
        array = array.T
    return array

def register_image(fixed_image, moving_image):
    """
    Register moving image to fixed image using SimpleITK with optimized parameters
    
    Args:
        fixed_image: SimpleITK Image or numpy array (reference image)
        moving_image: SimpleITK Image or numpy array (image to be registered)
        
    Returns:
        tuple: (registered_moving_image, translation_parameters)
               translation_parameters is [dx, dy] as floats (sub-pixel precision)
    """
    try:
        # Convert numpy arrays to SimpleITK images if needed
        if isinstance(fixed_image, np.ndarray):
            fixed_image = numpy_to_sitk_image(fixed_image)
        if isinstance(moving_image, np.ndarray):
            moving_image = numpy_to_sitk_image(moving_image)
        
        return _register_precise(fixed_image, moving_image)
        
    except Exception as e:
        print(f"Error registering image: {e}")
        return moving_image, [0.0, 0.0]

def _register_precise(fixed_image, moving_image):
    """Precise registration optimized for small displacements in hologram images"""
    
    # Preprocess images for better registration
    fixed_processed = _preprocess_for_registration(fixed_image)
    moving_processed = _preprocess_for_registration(moving_image)
    
    # Create registration method
    registration = sitk.ImageRegistrationMethod()
    
    # Use Mean Squares metric which is more sensitive to small changes
    registration.SetMetricAsMeanSquares()
    
    # Use Regular Step Gradient Descent for more precise control
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5,
        minStep=1e-4,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6
    )
    
    # Set optimizer scales for translation (equal weight for both directions)
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Set initial translation transform
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Use B-spline interpolation for sub-pixel accuracy
    registration.SetInterpolator(sitk.sitkBSpline)
    
    # Multi-level framework for better convergence
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    transform = registration.Execute(fixed_processed, moving_processed)
    parameters = list(transform.GetParameters())
    
    # Apply transform with high-quality interpolation
    registered_moving = _apply_transform_with_padding(moving_image, transform, fixed_image)
    
    return registered_moving, parameters

def _preprocess_for_registration(image):
    """Preprocess image to enhance registration quality"""
    # Apply slight Gaussian smoothing to reduce noise
    smoothed = sitk.SmoothingRecursiveGaussian(image, sigma=0.5)
    
    # Enhance edges using gradient magnitude
    gradient = sitk.GradientMagnitude(smoothed)
    
    # Combine original and gradient information
    enhanced = sitk.Add(sitk.Multiply(smoothed, 0.7), sitk.Multiply(gradient, 0.3))
    
    return enhanced

def register_images(data):
    """
    Register multiple images with the first image as reference
    
    Args:
        data: numpy array of shape (num_images, rows, cols)
        
    Returns:
        tuple: (registered_data, translations)
               registered_data: numpy array with registered images
               translations: list of [dx, dy] translation parameters for each image
    """
    try:
        num_images = data.shape[0]
                
        # Convert to list of SimpleITK images
        holo_images = []
        for i in range(num_images):
            image = numpy_to_sitk_image(data[i])
            holo_images.append(image)
        
        # Initialize translations array
        translations = [[0.0, 0.0] for _ in range(num_images)]
        
        # Register each image to the first one
        for i in range(1, num_images):
            registered_img, translation = register_image(holo_images[0], holo_images[i])
            holo_images[i] = registered_img
            translations[i] = translation
        
        # Convert back to numpy array
        registered_data = np.zeros_like(data)
        for i in range(num_images):
            registered_data[i] = sitk_image_to_numpy(holo_images[i])
        
        return registered_data, translations
        
    except Exception as e:
        print(f"Error registering images: {e}")
        return data, [[0.0, 0.0] for _ in range(num_images)]

def _apply_transform_with_padding(moving_image, transform, reference_image):
    """Apply transform with proper padding and extraction
    Uses integer pixel shifts to preserve original pixel values
    """
    
    # Get original sub-pixel parameters for padding calculation
    parameters = transform.GetParameters()
    
    pad_bound = [max(1, int(np.ceil(abs(parameters[0])))), 
                 max(1, int(np.ceil(abs(parameters[1]))))]    
    padded_moving = sitk.ZeroFluxNeumannPad(moving_image, pad_bound, pad_bound)
    
    # Apply transform with nearest neighbor to preserve pixel values
    resampled = sitk.Resample(
        padded_moving,
        transform,
        sitk.sitkBSpline,
        0.0,
        padded_moving.GetPixelID()
    )
    
    # Extract the region of interest
    registered_moving = sitk.Extract(resampled, reference_image.GetSize(), pad_bound)
    
    return registered_moving

def apply_fixed_translations(data, translations):
    """
    Apply fixed translation values to images with sub-pixel precision
    Each image uses its own padding based on its translation
    
    Args:
        data: numpy array of shape (num_images, rows, cols)
        translations: list of [dx, dy] translation parameters for each image
                      first image translation should be [0, 0]
        
    Returns:
        numpy array: shifted images with same shape as input
    """
    try:
        if len(data.shape) != 3:
            raise ValueError(f"Data must be 3D array, got shape: {data.shape}")
        
        num_images, rows, cols = data.shape
        
        if len(translations) != num_images:
            raise ValueError(f"Number of translations ({len(translations)}) must match number of images ({num_images})")
        
        # Convert to SimpleITK images and process
        result_data = np.copy(data)
        original_size = [cols, rows]  # SimpleITK uses [width, height]
        
        for i in range(1, num_images):  # Skip first image (reference)
            dx, dy = translations[i]            
            # Skip if no movement needed
            if dx == 0.0 and dy == 0.0:
                print(f"    No movement needed, keeping original")
                continue
            
            # Convert to SimpleITK image
            sitk_image = numpy_to_sitk_image(data[i])
            
            # Calculate padding needed for displacement (ensure enough padding)
            pad_bound = [max(1, int(np.ceil(abs(dx)))), 
                         max(1, int(np.ceil(abs(dy))))]
            padded_image = sitk.ZeroFluxNeumannPad(sitk_image, pad_bound, pad_bound)
            
            # Create translation transform with sub-pixel parameters
            transform = sitk.TranslationTransform(2)
            transform.SetParameters([dx, dy])
            
            # Apply transform with B-spline interpolation for sub-pixel accuracy
            resampled = sitk.Resample(
                padded_image, 
                transform, 
                sitk.sitkBSpline,
                0.0,
                padded_image.GetPixelID()
            )
            
            # Extract original size region
            extracted = sitk.Extract(resampled, original_size, pad_bound)
            
            # Convert back to numpy
            result_data[i] = sitk_image_to_numpy(extracted)
        
        return result_data
        
    except Exception as e:
        print(f"Error applying fixed translations: {e}")
        import traceback
        traceback.print_exc()
        return data