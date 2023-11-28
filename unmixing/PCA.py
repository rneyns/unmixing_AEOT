# Import packages
import os
from osgeo import gdal
import numpy
from sklearn.decomposition import PCA

# Define input and parameters --> replace PATH with the actual path to the file
def extract_PCA(InputImagery, NumberComponents):

  # Read imagery
  Image = gdal.Open(InputImagery).ReadAsArray()
  
  # Get imagery information
  prj = gdal.Open(InputImagery).GetProjectionRef()
  extent = gdal.Open(InputImagery).GetGeoTransform()
  
  # Adapt shape from (bands, row, col) to (row, col,bands)
  Image_AdaptedAxisOrder = numpy.moveaxis(Image,0,-1)
  
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

  # Write output
  for i in range (0,NumberComponents):
      OutputArray = pca_output_reshape[:,:,i]
      
      # Write output
      OutputNamePath = os.path.join(OutputFolder,'PCA_PC_'+str(i+1)+'.tif')
      Output = gdal.GetDriverByName('GTiff').Create(OutputNamePath,Image.shape[2],Image.shape[1],1,gdal.GDT_Float64)
      Output.SetProjection(prj)
      Output.SetGeoTransform(extent)
      Output.GetRasterBand(1).WriteArray(OutputArray)
      Output.GetRasterBand(1).SetNoDataValue(-9999)
      del Output
  
  # Return the resulting image as a HSI CUBE
  return pca_output.reshape((Image.shape[2],Image.shape[1],NumberComponents))
