import xarray as xr
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

class evve_orographic:

    def __init__(self, ds, elev_var=None):
        '''
        Class for orographic updraft computations using two different models:
        
				  'evve': Orographic estimation according to Thedin et al. (2023)
          'bo04': Orographic estimation according to Brandes and Ombalski (2004)

        - Thedin, R, Brandes, D, Quon, E, Sandhu, R, Tripp, C. "A three-dimensional model
        of terrain-induced updrafts for movement ecology studies". Under review, Journal
        of Ecological Applications, 2023.

        - Brandes, D., & Ombalski, D. W. "Modeling raptor migration pathways
        using a fluid-flow analogy". Journal of Raptor Research, v. 38, 2004 

        Inputs:
        -------
        ds: xr.Dataset
            Elevation map in xarray format. Coordinates should be x and y.
            Constant resolution is required. If xr.Dataset with multiple variables,
            see elev_var input
        elev_var: str
            If the ds is a xr.Dataset with multiple data variables, the variable that
            contains the elevation map should be given.

        '''
        self.ds = ds.copy()
        self.elev_var = elev_var
        self.orog_computed = False

        # Check ds input
        self._check_inputs(initialCheck=True)
        

    def __repr__(self):
        s  = f"EVVE object for orographic updraft estimation\n"
        s += f"  Elevation map variable: {self.elev_var}\n"
        s += f"  Resolution: {self.res} m\n"
        s += f"  Domain extents: x = [{min(self.ds['x'].values)}, {max(self.ds['x'].values)}] m\n"
        s += f"                  y = [{min(self.ds['y'].values)}, {max(self.ds['y'].values)}] m\n\n"
        if self.orog_computed:
            s += f"  Orographic updraft computed using the '{self.model}' model with the following conditions\n"
            s += f"    Wind direction: {self.wdir} degrees\n"
            s += f"    Height of interest: {self.h} m\n"
            if self.model == 'evve':
                s += f"    Wind speed at the reference height of 80 m AGL: {self.wspdAtRef:.2f} m/s\n"
            else:
                s += f"    Wind speed at the height of interest: {self.wspdAtHeightH:.2f} m/s \n"
            s += f"\n  Terrain properties and orographic updraft are\n"
            s += f"  available through the `<evve_object>.ds`.\n"
        else:
            s += f"\n  Orographic updraft not computed. To compute it, call\n"
            s += f"  .calcOrographicUpdraft( <see docstrings for inputs> )\n"
        return s


    def calcOrographicUpdraft (self, wdir, h, wspdAtRef=None, wspdAtHeightH=None, model=None):
        '''
        Calculates the orographic updraft according to the the model requested.
				The options are:
				'evve': Orographic estimation according to Thedin et al. (2023)
        'bo04': Orographic estimation according to Brandes and Ombalski (2004)

        Inputs:
        -------
        wdir: scalar
            Wind direction in typical convention where W = 270 deg.
        h: scalar
            Height of interest, in m.
        wspdAtRef: scalar
            Wind speed at the reference height (80 m), in m/s, regarless of
            the height of interest. Source of this quantity can be datasets
            such as NREL's WIND ToolKit or NOAA's HRRR.
            Only used for evve model
        wspdAtHeightH: scalar
            Wind speed at the desired height h, in m/s.
            Only used for bo04 model
        model: str
            Orographic updraft model. Options: `evve` or `bo04`

        '''

        # Check the new inputs
        self.wdir          = wdir
        self.h             = h
        self.wspdAtRef     = wspdAtRef
        self.wspdAtHeightH = wspdAtHeightH
        self.model         = model
        self._check_inputs()

        print(f'Computing the orographic updraft using {self.model.upper()} model')

        if model == 'evve':
            self.updraft_var = 'w0_evve'
            self._calc_orographic_updraft_evve()
        else:
            self.updraft_var = 'w0_bo04'    
            self._calc_orographic_updraft_bo04()

        self.orog_computed = True
        print('Done.')

        return


    def _calc_orographic_updraft_evve(self):
        '''
        Calculates the orographic updraft according to Thedin et al. (2023)

        Thedin, R, Brandes, D, Quon, E, Sandhu, R, Tripp, C. "A three-dimensional model
        of terrain-induced updrafts for movement ecology studies". Under review, Journal
        of Ecological Applications, 2023.
        '''

        # Get blurred elevation array
        self.elev = self._blur_quantity(self.ds[self.elev_var])

        # Get slope and aspect based on the elevation array above
        self._calc_slope_degrees()
        self._calc_aspect_degrees()

        # Get adjustment factors
        f_h  = self._evve_adjustment_height()       # Eq. (2)
        f_sx = self._evve_adjustment_sx()           # Eq. (5)
        f_tc = self._evve_adjustment_tc()           # Eq. (8)
        F = f_sx * f_tc / f_h                       # Eq. (9)

        # Estimate the updraft using the EVVE model
        sinterm = np.sin(np.deg2rad(self.ds['slope']))
        costerm = np.cos(np.deg2rad(self.ds['aspect'] - self.wdir))
        W0prime = sinterm*costerm
        w0i = self.wspdAtRef * F * W0prime          # Eq. (10)

        # Remove NaNs from last column and row (result of Sx calculation)
        w0i = w0i.values
        w0i[np.isnan(w0i)] = 0.0

        # Add final updraft estimation to dataset
        self.ds[self.updraft_var] = (('x','y'), w0i)


    def _calc_orographic_updraft_bo04(self):
        '''
        Calculates the orographic updraft according to Brandes and Ombalski (2004)

        Brandes, D., & Ombalski, D. W. "Modeling raptor migration pathways
        using a fluid-flow analogy". Journal of Raptor Research, v. 38, 2004 
        '''

        # Get elevation array (non-blurred, original)
        self.elev = self.ds[self.elev_var].values

        # Get slope and aspect based on the elevation array above
        self._calc_slope_degrees()
        self._calc_aspect_degrees()

        sinterm = np.sin(np.deg2rad(self.ds['slope']))
        costerm = np.cos(np.deg2rad(self.ds['aspect'] - self.wdir))
        w0 = self.wspdAtHeightH * sinterm * costerm

        # Add final updraft estimation to dataset
        self.ds[self.updraft_var] = (('x','y'), w0.values)
        

    def _evve_adjustment_height(self):
        '''       
        Compute height adjustment for the EVVE orographic model
        '''

        print('  Computing height adjustment factor from the EVVE model (1/3)..')
        a=0.00004;  b=0.0028;   c=0.8;
        d=0.35;     e=0.095;    f=-0.09
                
        slope = self.ds['slope']

        factor_height = ( a*self.h**2 + b*self.h + c ) * \
                        d**(-np.cos(np.deg2rad(slope)) + e) + f    # Eq. (2)

        return factor_height


    def _evve_adjustment_sx(self):
        '''
        Compute Sx adjustment for the EVVE orographic model
        '''

        print('  Computing Sx adjustment factor from the EVVE model (2/3)..')
        self._calc_sx()
        factor_sx = 1 + np.tan(np.deg2rad(self.ds['sx']))

        return factor_sx


    def _evve_adjustment_tc(self):
        '''
        Compute terrain complexity adjustment for the EVVE orographic model
        '''

        print('  Computing terrain complexity adjustment factor from the EVVE model (3/3)..')
        filterSize_in_m = 500

        elev = self.ds[self.elev_var]
        filterSize = int(np.floor(filterSize_in_m/self.res))

        local_zmean = ndimage.generic_filter(elev, np.mean, footprint=np.ones((filterSize,filterSize)) )
        local_zmin  = ndimage.generic_filter(elev, np.min,  footprint=np.ones((filterSize,filterSize)) )
        local_zmax  = ndimage.generic_filter(elev, np.max,  footprint=np.ones((filterSize,filterSize)) )

        tc = (local_zmean - local_zmin) / (local_zmax - local_zmin)    # Eq. (7)
        factor_tc = 1 + (self.h/40)*tc                                 # Eq. (8)

        return factor_tc


    def applyThreshold(self, min_updraft_value, inplace=True):
        '''
        Apply threshold on the underlying updraft value

        Input:
        ------
        min_updraft_value: scalar
          Minimum value to appear on the orographic updraft. Everything below this
          threshold will be set to zero.
        inplace: bool
          Whether or not to calculate the thresholded value in place
        '''

        if not isinstance(min_updraft_value,(int,float)):
            raise ValueError (f'The minimum updraft value needs to be given as a scalar')

        w0_threshold = xr.where(self.ds[self.updraft_var] < min_updraft_value, 0, self.ds[self.updraft_var])
       
        if inplace:
            self.ds[self.updraft_var] = w0_threshold
        else:
            #thresh_var = self.updraft_var + 
            self.ds[self.updraft_var+'_threshold'] = w0_threshold
        


    def _check_inputs(self, initialCheck=False):
        '''
        Perform some checks on the input data and inform the user
        of data requirements
        '''

        if not isinstance(self.ds, xr.Dataset):
            print(f'The elevation map should be given in xarray format')

        varlist = self.ds.keys()
        nvars = len(varlist)
        if nvars != 1 and self.elev_var is None:
            raise ValueError(f'The dataset given contains more than one data variable. To '\
                             f'specify the elevation variable, use elev_var=<your_elev_var_name>.')
        elif nvars == 1:
            self.elev_var = list(self.ds.keys())[0]
        else:
            # Dataset has many variables and elevation variable was given
            if self.elev_var not in varlist:
                raise ValueError (f'Requested elevation variable {self.elev_var} does not exist. '\
                                  f'Available options are {varlist}.')

        if 'x' not in list(self.ds.coords):
            raise ValueError(f"x should be a coordinate in the elevation array. "\
                             f"Current coordinates are {list(self.ds.coords)}")
        if 'y' not in list(self.ds.coords):
            raise ValueError(f"y should be a coordinate in the elevation array. "\
                             f"Current coordinates are {list(self.ds.coords)}")

        if len(np.unique(np.diff(self.ds.x))) != 1:
            raise ValueError(f"The resolution should be connstant in x")
        if len(np.unique(np.diff(self.ds.y))) != 1:
            raise ValueError(f"The resolution should be connstant in y")
        self._determine_resolution()


        if initialCheck is True:
            return


        if not isinstance(self.wdir, (float,int)):
            raise ValueError (f'The wind direction should be a scalar')
        if self.wdir<0 or self.wdir>360:
            print(f'! WARNING: Expecting wind direction to be given between 0 and '\
                  f'360 degrees. Received {wdir}. Assuming that is {wdir%360}.')
            self.wdir = self.wdir%360

        if not isinstance(self.h, (float,int)):
            raise ValueError(f'The desired height should be a scalar')

        if self.model == 'evve':
            if self.wspdAtRef is None:
                raise ValueError (f'For the EVVE model, the wind speed at the '\
                                  f'reference height `wspdAtRef` should be given')
            if self.wspdAtHeightH is not None:
                raise ValueError (f'EVVE model was requested. Wind speed at desired '\
                                  f'height should not be given.')
            if not isinstance(self.wspdAtRef, (float,int)):
                raise ValueError(f'The reference wind speed should be a scalar')
            if self.wspdAtRef > 15:
                print(f"! WARNING: Estimates of orographic updraft under high wind "\
                      f"conditions on the leeward side are not reliable.")
            if self.h>200 or self.h<30:
                print(f"! WARNING: The model's best performing range of height"\
                       " is between 30 and 200 m.")

        elif self.model == 'bo04':
            if self.wspdAtHeightH is None:
                raise ValueError (f'For the BO04 model, the wind speed at the desired'\
                                  f' height {self.h} `wspdAtHeightH` should be given.')
            if self.wspdAtRef is not None:
                raise ValueError (f'BO04 model was requested. Wind speed at reference '\
                                  f'height should not be given.')
            if not isinstance(self.wspdAtHeightH, (float,int)):
                raise ValueError(f'The wind speed given for height {self.h} should be a scalar')

        else:
            raise ValueError(f'Model can only be `evve` or `bo04`. Received {self.model}.')


    def _determine_resolution(self):
        '''
        Caclulate the resolution. Elevation map is assumed of uniform resolution
        '''
        self.res = (self.ds.x[1]-self.ds.x[0]).values


    def _calc_slope_degrees(self):
        '''
        Calculate local terrain slope using 3x3 stencil
        '''

        z = self.elev

        slope = np.empty_like(z)
        slope[:,:] = np.nan
        z1 = z[  :-2, 2:  ]  # upper left
        z2 = z[ 1:-1, 2:  ]  # upper middle
        z3 = z[ 2:  , 2:  ]  # upper right
        z4 = z[  :-2, 1:-1]  # center left
       #z5 = z[ 1:-1, 1:-1]  # center
        z6 = z[ 2:  , 1:-1]  # center right
        z7 = z[  :-2,  :-2]  # lower left
        z8 = z[ 1:-1,  :-2]  # lower middle
        z9 = z[ 2:  ,  :-2]  # lower right
        dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8*self.res)
        dz_dy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8*self.res)
        rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
        slope[1:-1,1:-1] = np.degrees(np.arctan(rise_run))

        # Add results to main dataset
        self.ds['slope'] = (('x','y'), slope)


    def _calc_aspect_degrees(self):
        '''
        Calculate local terrain aspect using 3x3 stencil
        '''

        z = self.elev

        aspect = np.empty_like(z)
        aspect[:, :] = np.nan
        z1 = z[  :-2, 2:  ]  # upper left
        z2 = z[ 1:-1, 2:  ]  # upper middle
        z3 = z[ 2:  , 2:  ]  # upper right
        z4 = z[  :-2, 1:-1]  # center left
       #z5 = z[ 1:-1, 1:-1]  # center
        z6 = z[ 2:  , 1:-1]  # center right
        z7 = z[  :-2,  :-2]  # lower left
        z8 = z[ 1:-1,  :-2]  # lower middle
        z9 = z[ 2:  ,  :-2]  # lower right
        dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8*self.res)
        dz_dy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8*self.res)
        dz_dx[dz_dx == 0.] = 1e-10
        angle = np.degrees(np.arctan(np.divide(dz_dy, dz_dx)))
        angle_mod = 90. * np.divide(dz_dx, np.absolute(dz_dx))
        aspect[1:-1, 1:-1] = 180. - angle + angle_mod

        # Change reference
        #aspect[1:-1, 1:-1] = (-aspect[1:-1, 1:-1]+90)%360

        # Add results to main dataset
        self.ds['aspect'] = (('x','y'), aspect)


    def _blur_quantity(self, qoi):
        '''
        Calculate a blurred version of a quantity quant based
        on the height h
        '''

        sigma_in_m = min(0.8*self.h + 16, 300) # size of kernel in meters, Eq. (6)
        return ndimage.gaussian_filter(qoi, sigma=sigma_in_m/self.res)


    def _calc_sx(self, dmax=500):
        '''
        Sx is a measure of topographic shelter or exposure relative to a particular
        wind direction. Calculates a whole map for all points (xi, yi) in the domain.
        For each (xi, yi) pair, it uses all v points (xv, yv) upwind of (xi, yi) in
        the A wind direction, up to dmax.

        Winstral, A., Marks D. "Simulating wind fields and snow redistribution using
            terrain-based parameters to model snow accumulation and melt over a semi-
            arid mountain catchment" Hydrol. Process. 16, 3585–3603 (2002)
        Input
        =====
        dmax: float
            Upwind extent of the search
        '''
        
        xx, yy = np.meshgrid(self.ds['x'], self.ds['y'], indexing='ij')

        # get resolution (assumes uniform resolution)
        npoints = 1+int(dmax/self.res)
        if dmax < self.res:
            raise ValueError('dmax needs to be larger or equal to the resolution of the grid')
        
        # Get upstream direction
        if    self.wdir==0:                     upstreamDirX=0;  upstreamDirY=-1
        elif  self.wdir==90:                    upstreamDirX=-1; upstreamDirY=0
        elif  self.wdir==180:                   upstreamDirX=0;  upstreamDirY=1
        elif  self.wdir==270:                   upstreamDirX=1;  upstreamDirY=0
        elif  self.wdir>0   and self.wdir<90:   upstreamDirX=-1; upstreamDirY=-1
        elif  self.wdir>90  and self.wdir<180:  upstreamDirX=-1; upstreamDirY=1
        elif  self.wdir>180 and self.wdir<270:  upstreamDirX=1;  upstreamDirY=1
        elif  self.wdir>270 and self.wdir<360:  upstreamDirX=1;  upstreamDirY=-1

        # change angle notation
        ang = np.deg2rad(270-self.wdir)

        # array for interpolation using griddata
        points = np.array( (xx.flatten(), yy.flatten()) ).T
        zagl = self.ds[self.elev_var].values
        values = zagl.flatten()

        # create rotated grid. This way we sample into a interpolated grid that has the exact points we need
        xmin = min(self.ds['x']);  xmax = max(self.ds['x'])
        ymin = min(self.ds['y']);  ymax = max(self.ds['y'])
        if self.wdir%90 == 0:
            # if flow is aligned, we don't need a new grid
            xrot = xx[:,0]
            yrot = yy[0,:]
            xxrot = xx
            yyrot = yy
            elevrot = zagl
        else:
            xrot = np.arange(xmin, xmax+0.1, abs(res*np.cos(ang)))
            yrot = np.arange(ymin, ymax+0.1, abs(res*np.sin(ang)))
            xxrot, yyrot = np.meshgrid(xrot, yrot, indexing='ij')
            elevrot = griddata( points, values, (xxrot, yyrot), method='linear' )

        # create empty rotated Sx array
        Sxrot = np.empty(np.shape(elevrot));  Sxrot[:,:] = np.nan

        for i, xi in enumerate(xrot):
            print(f'  Computing shelter angle Sx.. {100*(i+1)/len(xrot):.1f}%  ', end='\r')
            for j, yi in enumerate(yrot):

                # Get elevation profile along the direction asked
                isel = np.linspace(i-upstreamDirX*npoints+upstreamDirX, i, npoints, dtype=int)
                jsel = np.linspace(j-upstreamDirY*npoints+upstreamDirY, j, npoints, dtype=int)
                try:
                    xsel = xrot[isel]
                    ysel = yrot[jsel]
                    elev = elevrot[isel,jsel]
                except IndexError:
                    # At the borders, can't get a valid positions
                    xsel = np.zeros(np.size(isel))  
                    ysel = np.zeros(np.size(jsel))
                    elev = np.zeros(np.size(isel))

                # elevation of (xi, yi), for convenience
                elevi = elev[-1]

                try:
                    Sxrot[i,j] = np.nanmax(np.rad2deg( np.arctan( (elev[:-1] - elevi)/(((xsel[:-1]-xi)**2 + (ysel[:-1]-yi)**2)**0.5) ) ))
                except IndexError:
                    raise

        print(f'  Computing shelter angle Sx..        ', end='\r')
        # interpolate results back to original grid
        pointsrot = np.array( (xxrot.flatten(), yyrot.flatten()) ).T
        Sx = griddata( pointsrot, Sxrot.flatten(), (xx, yy), method='linear' )

        # Add results to main dataset
        self.ds['sx'] = (('x','y'), Sx)

        return Sx


    def plotTerrain(self, cropEdges=True):
        '''
        Plot the terrain elevation, and, if available, the slope and aspect.

        For the EVVE model, the slope and aspect that are available are not the actual, high-resolution
        map of such quantity, but rather a smoothed-out version used in the calculation of the updraft.
        Because of that, if your model is EVVE, the plots of slope and aspect will not be shown. If you
        wish to obtain the slope and aspect map, obtain a `bo04` version of the updraft and the slope &
        aspect fields will be available.

        Inputs
        ------
        cropEdges: bool
            Crop a 500 m fringe on each side for plotting purposes only.
        '''

        nplots = 1
        plotslope = plotaspect = False
        if self.model == 'bo04':
            if 'slope' in list(self.ds.keys()):
                nplots += 1
                plotslope=True
            if 'aspect' in list(self.ds.keys()):
                nplots += 1
                plotaspect=True

        xx, yy = np.meshgrid(self.ds['x'], self.ds['y'], indexing='ij')

        fig, axs = plt.subplots(ncols=nplots, nrows=1, figsize=(5*nplots,4), gridspec_kw={'wspace':0.40,'hspace':0.1})
        if nplots==1: axs = [axs]
        
        props = dict(facecolor='white', alpha=0.6, edgecolor='silver', boxstyle='square', pad=0.15)

        cm = axs[0].pcolormesh(xx, yy, self.ds[self.elev_var], cmap='terrain', shading='auto', rasterized=True)#,vmin=0,vmax=0.003)
        cb = fig.colorbar(cm,ax=axs[0], label='Relative elevation [m]', fraction=0.046)
        axs[0].text(0.97, 0.97, r'$(a)$', color='black', ha='right', va='top', transform=axs[0].transAxes, fontsize=14, bbox=props)
        
        if plotslope:
            cm = axs[1].pcolormesh(xx, yy, self.ds['slope'], cmap='cividis', shading='auto',vmin=0, rasterized=True)
            cb = fig.colorbar(cm,ax=axs[1], label='Slope [deg]', fraction=0.046)
            axs[1].text(0.97, 0.97, r'$(b)$', color='black', ha='right', va='top', transform=axs[1].transAxes, fontsize=14, bbox=props)
        
        if plotaspect:
            cm = axs[2].pcolormesh(xx, yy, self.ds['aspect'], cmap='twilight', shading='auto',vmin=0,vmax=360, rasterized=True)
            cb = fig.colorbar(cm,ax=axs[2], label='Aspect',  ticks=[0,45,90,135,180,225,270,315,360], fraction=0.046)
            cb.ax.set_yticklabels(['N','','E','','S','','W','','N'])
            axs[2].text(0.97, 0.97, r'$(c)$', color='black', ha='right', va='top', transform=axs[2].transAxes, fontsize=14, bbox=props)
        
        for ax in axs:
        		ax.set_aspect('equal');
        		if cropEdges:
        		    ax.set_xlim([min(self.ds.x)+505, max(self.ds.x)-515])
        		    ax.set_ylim([min(self.ds.y)+505, max(self.ds.y)-505])
        
        plt.show()

    
    def plotOrographicUpdraft(self, cropEdges=True, **kwargs):
        '''
        Plot the final orograhic updraft.

        Inputs
        ------
        cropEdges: bool
            Crop a 500 m fringe on each side for plotting purposes only.
        kwargs:
            Extra arguments that will be passed to the pcolormesh plotting function
            Useful available options are `cmap`, `vmin`, and `vmax`.
        '''

        xx, yy = np.meshgrid(self.ds['x'], self.ds['y'], indexing='ij')

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,4), gridspec_kw={'wspace':0.40,'hspace':0.1})
        
        cm = ax.pcolormesh(xx, yy, self.ds[self.updraft_var], shading='auto', **kwargs)
        cb = fig.colorbar(cm,ax=ax, label=f'Orographic updraft by {self.model.upper()} model [m/s]', fraction=0.046)
       
        ax.set_aspect('equal');
        if cropEdges:
       	    ax.set_xlim([min(self.ds.x)+505, max(self.ds.x)-515])
       	    ax.set_ylim([min(self.ds.y)+505, max(self.ds.y)-505])
        
        plt.show()







