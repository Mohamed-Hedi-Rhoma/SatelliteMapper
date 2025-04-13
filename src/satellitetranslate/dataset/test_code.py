import ee

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project="ee-get-landsat-data")

# Get first image from various collections
landsat8_l2 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').first()
landsat8_l1 = ee.ImageCollection('LANDSAT/LC08/C02/T1').first()
sentinel2_sr = ee.ImageCollection('COPERNICUS/S2_SR').first()
sentinel2_l1 = ee.ImageCollection('COPERNICUS/S2').first()

# Print band names
print("Landsat 8 L2 (SR) bands:")
print(landsat8_l2.bandNames().getInfo())
print("\nLandsat 8 L1 bands:")
print(landsat8_l1.bandNames().getInfo())
print("\nSentinel-2 L2A (SR) bands:")
print(sentinel2_sr.bandNames().getInfo())
print("\nSentinel-2 L1C bands:")
print(sentinel2_l1.bandNames().getInfo())

# Check for angle information
print("\nLandsat 8 L2 metadata properties:")
print(landsat8_l2.propertyNames().getInfo())
print("\nLandsat 8 L1 metadata properties:")
print(landsat8_l1.propertyNames().getInfo())
print("\nSentinel-2 L2A metadata properties:")
print(sentinel2_sr.propertyNames().getInfo())
print("\nSentinel-2 L1C metadata properties:")
print(sentinel2_l1.propertyNames().getInfo())

# Additional test to check if Sentinel-2 has angle bands
if 'MEAN_INCIDENCE_ZENITH_ANGLE_B2' in sentinel2_sr.bandNames().getInfo():
    print("\nSentinel-2 L2A has angle bands as bands")
else:
    print("\nSentinel-2 L2A does not have angle bands as bands")

if 'MEAN_INCIDENCE_ZENITH_ANGLE_B2' in sentinel2_l1.bandNames().getInfo():
    print("Sentinel-2 L1C has angle bands as bands")
else:
    print("Sentinel-2 L1C does not have angle bands as bands")

# Check if angles are in properties for Landsat
print("\nLandsat 8 L2 solar angles in properties:")
print("SUN_AZIMUTH:", landsat8_l2.get("SUN_AZIMUTH").getInfo())
print("SUN_ELEVATION:", landsat8_l2.get("SUN_ELEVATION").getInfo())

print("\nLandsat 8 L1 solar angles in properties:")
print("SUN_AZIMUTH:", landsat8_l1.get("SUN_AZIMUTH").getInfo())
print("SUN_ELEVATION:", landsat8_l1.get("SUN_ELEVATION").getInfo())