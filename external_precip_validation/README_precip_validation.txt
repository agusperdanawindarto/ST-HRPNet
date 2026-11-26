Independent precipitation validation options:
1. Save aligned binary masks as precip_test_masks_256.npy with shape (n_test, 256, 256, 1).
2. Save binary mask images under masks_by_time/<target_time_code>.png.
3. Save a grid CSV as precip_grid.csv with columns target_time_code, lon, lat, precip.
4. Save GPM/GSMaP NetCDF files under netcdf/ and configure PRECIP_VALUE_COLUMN if needed.
