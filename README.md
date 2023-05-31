# Engineering Vertical Velocity Estimator

This repository hosts models for updraft estimation for animal movement studies. 

## Orographic Updraft

An improved orographic updraft model based on that of Brandes and Ombalski (2004). The improved model is described in Thedin et al. (2023). 

The inputs are: digital elevation model (DEM) of the region of interest, the wind direction, the desired height, and the wind speed at a reference height of 80 m. The orographic updraft using the EVVE model can be obtained using

```python
from evve.orographic import orographic
import xarray as xr

# Open xarray Dataset containing the DEM information
ds = xr.open_dataset('examples/ds_WY.nc')  # see file for example format

# Initialize the object
evve_orog = orographic(ds)

# Estimate the orographic updraft using the EVVE model
evve_orog.calcOrographicUpdraft(wdir = 270,      # desired wind dir in typical wind convention
                                h = 120,         # height where the updrafts will be estimated
                                wspdAtRef = 8,   # wind speed at a reference height of 80 m AGL
                                model = 'evve'   # model of interest. Options are 'evve' or 'bo04'
                               )

# Plot the resulting orographic field
evve_orog.plotOrographicUpdraft(cmap='RdBu_r', vmin=-1, vmax=1)
```

Note that a more complete example is provided in the `examples` directory.


## Thermal Updraft

The thermal updraft model is a work in progress. It requires coupling to [HRRR](https://rapidrefresh.noaa.gov/hrrr/). The version available here is not in working condition yet. A working version, with the appropriate coupling to HRRR is available in the companion repository [SSRS](https://github.com/NREL/SSRS).


## References
- Thedin, R, Brandes, D, Quon, E, Sandhu, R, Tripp, C. "A three-dimensional model of terrain-induced updrafts for movement ecology studies". Under review, Journal of Movement Ecology, 2023.

- Brandes, D., & Ombalski, D. W. "Modeling raptor migration pathways using a fluid-flow analogy". Journal of Raptor Research, v. 38, 2004 
