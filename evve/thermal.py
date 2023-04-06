# ----------------------- #
# written by Regis Thedin
# regis.thedin@nrel.gov
# ----------------------- #

import numpy as np
import xarray as xr
import os
from datetime import datetime
import random
from scipy import ndimage
#from .hrrr import HRRR


class thermal:

    def __init__(self, ds):
        '''
        Class for thermal updraft computations.
        
        - Allen, M. "Updraft model for development of autonomous soaring uninhabited
        air vehicles." 44th AIAA Aerospace Sciences Meeting and Exhibit, 2006.
        
        Inputs:
        -------
        ds: xr.Dataset
            Elevation map in xarray format. Coordinates should be x and y.
            Constant resolution is required. If xr.Dataset with multiple variables,
            see elev_var input
        '''

        self.ds = ds.copy()
        self.time = time

        raise NotImplementedError('The class for thermal updrafts is not fully implemented'\
                                  ' yet. If needed, see the implementation directly on SSRS'\
                                  ' (https://github.com/NREL/SSRS)')



    def calcThermalUpdraft (self, aspect, southwest_lonlat, extent, res, time, height, wfipInformed = True):
        '''
        Returns field of thermals based on Allen (2006)

        Inputs:
        -------
        southwest_lonlat: Tuple[float, float] 
          Long and lat of the SW corner of the region of interes
        extent: Tuple[float, float, float, float]
          The extent of the desired region, given as [xmin, ymin, xmax, ymax]
        res: scalar
          resolution
        time:  tuple with [y, m, d, hour], or datetimeobject
          Time of interest (for HRRR)
        height: scalar
          Height of interest
        wfipInformed: bool
          Whether or not to use WFIP2 data to scale results. Default: True

        '''
        
        # TODO: Loop over a list of `time`s
        
        # Get string of time to pass to HRRR. Time can be passed either a tuple of datetime object
        if isinstance(time, datetime):
            timestr = f' {time.year}-{time.month:02d}-{time.day:02d} {time.hour:02d}:{time.minute:02d}'
        else:
            timestr = f'{time[0]}-{time[1]:02d}-{time[2]:02d} {time[3]:02d}:00'

        # Get hrrr data
        hrrr = HRRR(valid_date = timestr)

        # Compute convective velocity
        wstar,  xx, yy = hrrr.get_convective_velocity(southwest_lonlat, extent, res=res)
        # Compute albedo
        albedo, xx, yy = hrrr.get_albedo(southwest_lonlat, extent, res)
        # Get boundary layer height
        zi,     xx, yy = hrrr.get_single_var_on_grid(':(HPBL):',  # boundary layer height
                                                     southwest_lonlat,
                                                     extent,
                                                     res)
        wstar  = wstar.values
        try:
            albedo = albedo.values
        except AttributeError:  # it's an array already
            pass
        zi = zi[list(zi.keys())[0]].values


        if np.mean(zi) == np.nan:
            raise ValueError(f'The value obtained for the boundary layer height contains NaNs.',\
                             f'HRRR data is imcomplete at the site and time of interest.')

        # Define updraft shape factors
        r1r2sh = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
        ksh = np.array([[1.5352, 2.5826, -0.0113, -0.1950, 0.0008],
                        [1.5265, 3.6054, -0.0176, -0.1265, 0.0005],
                        [1.4866, 4.8356, -0.0320, -0.0818, 0.0001],
                        [1.2042, 7.7904,  0.0848, -0.0445, 0.0001],
                        [0.8816, 13.9720, 0.3404, -0.0216, 0.0001],
                        [0.7067, 23.9940, 0.5689, -0.0099, 0.0002],
                        [0.6189, 42.7965, 0.7157, -0.0033, 0.0001]])


        # Create weight for likeliness of thermals in space
        albedofactor = (0.1/(albedo)**0.5)
        spatialWeight = ( wstar**1 + albedofactor )**2
        # Mask the edges so no thermals there
        fringe= 2000 # in [m]
        ifringe = int(fringe/res)
        spatialWeight[0:ifringe,:] = spatialWeight[-ifringe:,:] = 0
        spatialWeight[:,0:ifringe] = spatialWeight[:,-ifringe:] = 0

        # Get thermal parameters
        ziavg = np.mean(zi)
        zzi = height/zi
        zziavg = height/ziavg
        if ziavg > 300:
            ValueError(f'The boundary layer is too shallow for thermals')

        # Calcualte average updraft size
        rbar=(.102*zzi**(1/3))*(1-(.25*zzi))*zi

        # Calculate average updraft strength (G. Young)
        wT = wstar * 0.85 * (zzi**(1/3)) * (1.3-zzi)

        # Size gain around a mean, based on albedo
        rgain =1.4*(0.4/(albedo))

        # Calculate inner and outer radius of rotated trapezoid updraft
        r2 = rbar*rgain;  r2[r2<10] = 10
        r1r2 = 0.0011*r2+0.14
        r1r2[r2>600] = 0.8
        r1 = r1r2*r2

        # Determine number of thermals
        nThermals = int ( 0.6*(extent[2]-extent[0])*(extent[3]-extent[1])/(ziavg*np.mean(r2)) )

        # Create strength gains, based on wstar
        wgain = 0.7*wstar

        # Multiply average updraft strength by the gain
        wTbar = wT*wgain

        # Calculate strength at center of rotated trapezoid updraft
        wpeak=(3*wTbar*((r2**3)-(r2**2)*r1)) / ((r2**3)-(r1**3))

        # Create a realization of thermal's center location
        print(f'Creating {nThermals} thermals. The average boundary layer height is {ziavg:.1f} m')
        wt_init, sumOfRealizations = self._get_random_points_weighted(weight=spatialWeight, n=nThermals, nRealization=1)

        # Get distances to closest thermal center
        wt_init1 = np.zeros_like(wt_init)
        wt_init1[wt_init>0]=1
        dist = ndimage.distance_transform_edt(np.logical_not(wt_init1)) * res

        # Calculate updraft velocity
        r=dist
        rr2=r/r2

        # Calculate shape parameters
        k1 = np.ones_like(r1r2)
        k2 = np.ones_like(r1r2)
        k3 = np.ones_like(r1r2)
        k4 = np.ones_like(r1r2)
        k1 = k1*ksh[6,0];                                k2 = k2*ksh[6,1];                                k3 = k3*ksh[6,2];                                      k4 = k4*ksh[6,3]
        k1[r1r2<(0.5*r1r2sh[6]+r1r2sh[5])] = ksh[5,0];   k2[r1r2<(0.5*r1r2sh[6]+r1r2sh[5])] = ksh[5,1];   k3[r1r2<(0.5*r1r2sh[6]+r1r2sh[5])] = ksh[5,2];   k4[r1r2<(0.5*r1r2sh[6]+r1r2sh[5])] = ksh[5,3]
        k1[r1r2<(0.5*r1r2sh[5]+r1r2sh[4])] = ksh[4,0];   k2[r1r2<(0.5*r1r2sh[5]+r1r2sh[4])] = ksh[4,1];   k3[r1r2<(0.5*r1r2sh[5]+r1r2sh[4])] = ksh[4,2];   k4[r1r2<(0.5*r1r2sh[5]+r1r2sh[4])] = ksh[4,3]
        k1[r1r2<(0.5*r1r2sh[4]+r1r2sh[3])] = ksh[3,0];   k2[r1r2<(0.5*r1r2sh[4]+r1r2sh[3])] = ksh[3,1];   k3[r1r2<(0.5*r1r2sh[4]+r1r2sh[3])] = ksh[3,2];   k4[r1r2<(0.5*r1r2sh[4]+r1r2sh[3])] = ksh[3,3]
        k1[r1r2<(0.5*r1r2sh[3]+r1r2sh[2])] = ksh[2,0];   k2[r1r2<(0.5*r1r2sh[3]+r1r2sh[2])] = ksh[2,1];   k3[r1r2<(0.5*r1r2sh[3]+r1r2sh[2])] = ksh[2,2];   k4[r1r2<(0.5*r1r2sh[3]+r1r2sh[2])] = ksh[2,3]
        k1[r1r2<(0.5*r1r2sh[2]+r1r2sh[1])] = ksh[1,0];   k2[r1r2<(0.5*r1r2sh[2]+r1r2sh[1])] = ksh[1,1];   k3[r1r2<(0.5*r1r2sh[2]+r1r2sh[1])] = ksh[1,2];   k4[r1r2<(0.5*r1r2sh[2]+r1r2sh[1])] = ksh[1,3]
        k1[r1r2<(0.5*r1r2sh[1]+r1r2sh[0])] = ksh[0,0];   k2[r1r2<(0.5*r1r2sh[1]+r1r2sh[0])] = ksh[0,1];   k3[r1r2<(0.5*r1r2sh[1]+r1r2sh[0])] = ksh[0,2];   k4[r1r2<(0.5*r1r2sh[1]+r1r2sh[0])] = ksh[0,3]

        # Calculate the smooth vertical velocity distribution
        ws = (1/(1+(k1*abs(rr2+k3))**k2)) + k4*rr2
        # no negative updrafts
        ws[ws<0] = 0
        # Set to zero if above the boundary layer
        ws[zi<height] = 0

        # Calculate downdraft velocity at edges of updrafts
        wl = (np.pi/6)*np.sin(rr2*np.pi)
        wl[( (dist<r1) | (rr2>2))] = 0
        wd = 2.5*wl*(zzi-0.5)
        wd[((zzi<0.5) | (zzi>0.9))] = 0
        wd[wd<0]=0

        # Combine fields
        w = wpeak*ws + wd*wTbar

        # Scale it to fit experimental data (optional)
        if wfipInformed:
            if height<= 200:
                wmax = self._get_obs_maxw(height, hrrr, southwest_lonlat, extent, res)
                w = w*wmax/np.max(w)
            else:
                print('The height requested is higher than observations. Skipping correction.')

        # Environment sink
        # we = np.zeros_like(w)
        # Stretch updraft field to blend with sink at edge
        # w[dist>r1] = (w*(1-we/wpeak)+we)[dist>r1]
        
        print(f'compute_thermals_3d returning a thermal field of shape {np.shape(w)}')
        return w


    def _get_random_points_weighted (weight, n, nRealization=1):
        
        normalweight = weight/np.sum(weight)
        
        choicesum = np.zeros_like(weight).flatten()
        for i in range(nRealization):
            randindices = np.random.choice(np.arange(np.size(choicesum)), size=n,
                                           replace = False, p=normalweight.flatten())
            # create a current-iteration result
            choice = np.zeros_like(weight).flatten()
            choice[randindices] = 1
            # accumulate
            choicesum = choicesum +choice

        # sum of realizations
        choicesum = choicesum.reshape(np.shape(weight))
        # last realization, choice
        choice = choice.reshape(np.shape(weight))
        
        return choice, choicesum


    def _get_obs_maxw(self, height, hrrr, southwest_lonlat, extent, res):
        # Add a weight based on experimental observations at the WFIP2 site if height is low

        if height<=200:
            my_path = os.path.abspath(os.path.dirname(__file__))
            wfip = xr.open_dataset(os.path.join(my_path,'updraft','updraft_conditions_wfip2.nc'))
            rho = 1.225 # kg/m^3
            cp = 1005   # J/(kg*K)

            # Get mean wspd
            u, xx, yy = hrrr.get_single_var_on_grid(':UGRD:80 m above ground',       southwest_lonlat, extent, res)   # u component of the wind at 80 AGL
            v, xx, yy = hrrr.get_single_var_on_grid(':UGRD:80 m above ground',       southwest_lonlat, extent, res)   # u component of the wind at 80 AGL
            wspd = (u**2+v**2)**0.5
            meanwspd = np.mean(wspd)

            # Get heat flux
            gflux_Wm2, xx, yy = hrrr.get_single_var_on_grid(':(GFLUX):',      southwest_lonlat, extent, res)   # ground heat flux
            sensible,  xx, yy = hrrr.get_single_var_on_grid(':SHTFL:surface', southwest_lonlat, extent, res)   # sensible heat flux
            latent,    xx, yy = hrrr.get_single_var_on_grid(':LHTFL:surface', southwest_lonlat, extent, res)   # latent heat flux
            hfx = (sensible + latent - gflux_Wm2 )/(rho*cp)
            meanhfx = np.mean(hfx)

            # Get the vertical speed statistics
            wfiph = wfip.interp(height=height).squeeze(drop=True)
            wdata = wfiph.where( (wfiph.wind_speed>meanwspd-1 ) & ( wfiph.wind_speed<meanwspd+1) &
                                 (wfiph.hfx>meanhfx-0.025 )     & ( wfiph.hfx<meanhfx+0.025),       drop=True )['vertical_air_velocity']#.to_dataframe().agg(['count','min','mean','max','std'])
            wmax = wdata.max().values
            
            return wmax



