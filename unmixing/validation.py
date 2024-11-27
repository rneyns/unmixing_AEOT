# Import packages
from osgeo import gdal
import numpy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from math import sqrt
from sklearn import metrics

MarkerSize = 1

# Functions --------------------------------------------------------------------
def SetValueNaN(Array):
    # Get value to set to nodata
    NoDataValue = Array[0,0]
    
    # Set value to nan
    Array[Array==NoDataValue]=numpy.nan
    
    return(Array)
    
def GetNoNaNValues(ArrayRef,ArrayPred):
    # Define empty array
    NoNaNValues_Ref = []
    NoNaNValues_Pred = []
    
    # Loop through pixels and get values that are not NaN
    for i in range(ArrayRef.shape[0]):
        for j in range(ArrayRef.shape[1]):
            if numpy.isnan(ArrayRef[i,j])==1 or numpy.isnan(ArrayPred[i,j])==1:
                pass
            else:
                NoNaNValues_Ref.append(ArrayRef[i,j])
                NoNaNValues_Pred.append(ArrayPred[i,j])
                
    return(NoNaNValues_Ref,NoNaNValues_Pred)
                
# Main script ------------------------------------------------------------------

def validation(Reference, Prediction, band_num):
  Ref = gdal.Open(Reference).ReadAsArray()
  Pred = gdal.Open(Prediction).ReadAsArray()
    
  if len(Pred.shape) > 2:
      Pred = Pred[band_num-1]
  
  print(Ref.shape)
  print(Pred.shape)
  
  # Check for shape
  if Ref.shape[0] == Pred.shape[0]:
      
      if Ref.shape[1] == Pred.shape[1]:
          
          # Correct array for no data
          Ref_NaN = SetValueNaN(Ref)
          Pred_NaN = SetValueNaN(Pred)
          
          # Get no nan values for error metrics
          Ref_ValidValues, Pred_ValidValues= GetNoNaNValues(Ref_NaN,Pred_NaN)
          
          # Calculate error metrics
          MAE = round(metrics.mean_absolute_error(Ref_ValidValues,Pred_ValidValues),2)
          RMSE = round(sqrt(metrics.mean_squared_error(Ref_ValidValues,Pred_ValidValues)),2)
          
          # Plot
          fig,ax = plt.subplots(figsize=(5,5))
          ax.axis('equal')
          ax.scatter(Ref_NaN,Pred_NaN,c = 'lightgrey',s=MarkerSize)
          ax.set_xlabel('Reference fraction', fontsize=12)
          ax.set_ylabel('Estimated fraction', fontsize=12)
          ax.set_ylim(-0.1,1.1)
          ax.set_xlim(-0.1,1.1)
          ax.annotate('MAE = '+str(MAE),xy=(0,0.95),fontsize=12)
          ax.annotate('RMSE = '+str(RMSE),xy=(0,0.85),fontsize=12)
          line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=1)
          
          ax.add_line(line)
          fig.show()
          
          print('Script finalized')
          
      else:
          print('Script stopped: Ref.shape[1] != Pred.shape[1]')
  
  else:
      print('Script stopped: Ref.shape[0] != Pred.shape[0]')
