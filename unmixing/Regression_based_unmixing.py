import numpy as np
import pandas as pd

def generate_synthetic_dataset_random_class_mixing(endmember_training_set, num_samples, classes):
    """
    Generate a synthetic dataset with mixed pixels for training unmixing algorithms.
    For each sample, draw a random number of endmembers from different classes.

    Parameters:
    - endmember_training_set: Dictionary where keys are class names, and values are 2D NumPy arrays representing
                              the endmember training set for each class (endmembers x bands).
    - num_samples: Number of synthetic samples to generate.
    - max_endmembers_per_class: Maximum number of endmembers to draw from each class for each sample.

    Returns:
    - synthetic_samples: 2D NumPy array representing the synthetic dataset (samples x bands).
    - class_labels: 1D NumPy array representing the class labels for each sample.
    """
    
    # Convert DataFrame to dictionary
    labels = endmember_training_set["VIS"]
    df = endmember_training_set.drop(["Desc","Bright","VIS"],axis=1)
    endmember_training_set = df.values

    # Initialize arrays to store the end-members synthetic samples and class labels
    synthetic_samples = []
    class_labels = []


    # Iterate over each sample
    for _ in range(num_samples):
        selected_endmembers = []
                
        #Choose the end-members that will be mixed
        for cl in classes:
            # Find indices where the value is equal to the target value
            indices = np.where(labels == cl)[0]
            random_index = np.random.choice(indices)
            selected_endmembers.append(endmember_training_set[random_index])
            
        # Generate a random abundance map
        abundance_maps = np.random.rand(len(classes))

        # Normalize abundances to sum to 1
        abundance_maps /= np.sum(abundance_maps)

        # Linear combination of selected endmembers
        mixed_spectrum = np.dot(abundance_maps, selected_endmembers)

        # Append synthetic sample and class label to the arrays
        synthetic_samples.append(mixed_spectrum)
        class_labels.append(abundance_maps)

    # Convert the lists to NumPy arrays
    synthetic_samples = np.array(synthetic_samples)
    class_labels = np.array(class_labels)

    return synthetic_samples, class_labels


def regression_based_unmixing(synthetic_samples,class_labels, hsi, classes=['Impervious','Vegetation','Soil']):
    
    """
    Train a classifier on a set of labeled mixed pixels and apply the classifier to the image to be unmixed

    Parameters:
    - synthetic_samples: The relectivity values of the samples on which the classifier should be trained
    - Class_labels: The labels associated with these synthetic samples (an array with a percentage of cover per class)
    - Hsi: the actual raster 
    - Classes: The array with classes present in the dataset, in the right order

    Returns:
    - The unmixed raster in an hsi data format
    """

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(synthetic_samples, class_labels, test_size=0.1, random_state=42)
    
    # Initialize and train the Random Forest model with multi-output regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    multi_output_rf_model = MultiOutputRegressor(rf_model)
    multi_output_rf_model.fit(X_train, y_train)
    
    # Predict class probabilities on the test set
    y_pred = multi_output_rf_model.predict(X_test)
    
    # Evaluate the model performance
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae}')
    
    # Adapt shape from (bands, row, col) to (row, col,bands)
    Image_AdaptedAxisOrder =  np.moveaxis(hsi,0,-1)
    
    # Adapt shape from 3D to 2D; new shape = (row * col, bands)
    Image_2D = Image_AdaptedAxisOrder[:, :, :].reshape((hsi.shape[1] * hsi.shape[2], hsi.shape[0]))
    
    #Apply the algorithm 
    unmixed_raster = multi_output_rf_model.predict(Image_2D)
    
    # Reshape the output
    unmixed_reshaped = unmixed_raster.reshape((hsi.shape[1], hsi.shape[2], len(classes)))

    # Return the resulting image as a HSI CUBE
    return np.transpose(unmixed_reshaped, (1,0, 2))
