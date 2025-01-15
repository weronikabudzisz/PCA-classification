import rasterio
import os
import numpy as np
import geopandas as gpd
from sklearn.decomposition import PCA
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the stack_ndvi raster
ndvi = 'NDVI.tif'
with rasterio.open(ndvi) as src:
    bands = src.read()
    profile = src.profile
    transform = src.transform

band1 = bands[0]  
band2 = bands[1]
band3 = bands[2]

# Calculate correlation coefficient
correlation = np.corrcoef(band2.flatten(), band3.flatten())[0, 1]
print(f"Correlation between band 2 and band 3: {correlation}")

# Reshape the raster stack for PCA (flatten spatial dimensions)
n_bands, n_rows, n_cols = bands.shape
data = bands.reshape(n_bands, -1).T  # Shape: (pixels, bands)

# Perform PCA
pca = PCA(n_components=10)
pca_result = pca.fit_transform(data)

# Reshape back to raster dimensions
pca_rasters = pca_result.T.reshape(10, n_rows, n_cols)

# Display the first 3 principal components
# for i in range(3):
#     plt.figure(figsize=(6, 6))
#     plt.title(f'Principal Component {i+1}')
#     plt.imshow(pca_rasters[i], cmap='gray')
#     plt.colorbar()
#     plt.show()

# Normalize PC1, PC2, and PC3 for visualization
def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

pc1 = normalize(pca_rasters[0])
pc2 = normalize(pca_rasters[1])
pc3 = normalize(pca_rasters[2])

# Create RGB image
rgb_image = np.stack([pc1, pc2, pc3], axis=-1)

# Display RGB
# plt.figure(figsize=(6, 6))
# plt.title('RGB Composite from PCA')
# plt.imshow(rgb_image)
# plt.show()

#correlation matrix
b1 = pca_rasters[0].flatten()  # Flatten PC1
b2 = pca_rasters[1].flatten()  # Flatten PC2
b3 = pca_rasters[2].flatten()  # Flatten PC3

channels = np.vstack((band1.flatten(), band2.flatten(), band3.flatten()))

corr_matrix = np.corrcoef(channels)
print("\nCorrelation Matrix:")
print(corr_matrix)

cov_matrix = np.cov(channels)
print("\nCovariance Matrix:")
print(cov_matrix)

# Transpose the array to (3, height, width) for writing with rasterio
rgb_image_transposed = rgb_image.transpose(2, 0, 1)  # (height, width, 3) -> (3, height, width)

# Define output directory and file path
output_dir = r'C:\Users\Weronika\Desktop'
output_filename = 'rgb_image.tif'
output_path = os.path.join(output_dir, output_filename)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get the transform and CRS from the original raster (NDVI.tif)
with rasterio.open('NDVI.tif') as src:
    transform = src.transform
    crs = src.crs

# Write the RGB image to a new TIFF file
with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    count=3,  # Three bands (RGB)
    dtype=rgb_image_transposed.dtype,
    width=rgb_image_transposed.shape[2],
    height=rgb_image_transposed.shape[1],
    crs=crs,  # Coordinate Reference System (from original raster)
    transform=transform  # Affine transform for spatial referencing
) as dst:
    for i in range(3):  # Write each RGB band to the file
        dst.write(rgb_image_transposed[i], i+1)

print(f"RGB image exported to {output_path}")