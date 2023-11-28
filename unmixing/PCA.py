# Import packages
import os
from osgeo import gdal
import numpy
from sklearn.decomposition import PCA

# Define input and parameters --> replace PATH with the actual path to the file
InputImagery = r"PATH"
OutputFolder = r"PATH"
NumberComponents = 3
def extract_PCA(InputImagery, NumberComponents):
  
  # Adapt shape from 3D to 2D; new shape = (row * col, bands)
  Image_2D = Image_AdaptedAxisOrder[:, :, :].reshape((Image.shape[1] * Image.shape[2],Image.shape[0]))
  
  # Fit PCA
  pca=PCA(n_components=NumberComponents)
  pca.fit(Image_2D)
  
  # Write out PCA information
  print('Explained variance by the first '+str(NumberComponents)+' principal components: \n')
  print(str(pca.explained_variance_ratio_)+'\n')
  print('\nTotal explained variance by the first '+str(NumberComponents)+' principal components: \n' )
  print(str(round(numpy.sum(pca.explained_variance_ratio_),2))+'\n')
  print('\nComponent matrix: \n')
  print(str(pca.components_))
  print()
  
  # Apply PCA
  pca_output = pca.fit_transform(Image_2D)
  
  # Reshape PCA output
  pca_output_reshape = pca_output.reshape((Image.shape[1],Image.shape[2],NumberComponents))
  
  # Return the resulting image
  return pca_output_reshape
