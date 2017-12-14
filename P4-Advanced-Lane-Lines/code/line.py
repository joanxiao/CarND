# Define a class to receive the characteristics of each line detection
import collections
import numpy as np

class Line():
    def __init__(self, curverad_factor=10):
        #the maximum value of current/new curverad or new/current curverad can best_fit
        if curverad_factor == 0:
            curverad_factor = 1
        self.curverad_factor = curverad_factor
        # was the line detected in the last iteration?
        self.detected = False    
        #polynomial coefficients averaged over the last 10 iterations
        self.best_fit = None          
        #radius of curvature of the line in meters
        self.curverad = 0 
        #polynomial coefficients of the last 10 fits
        self.recent_fits = collections.deque(maxlen=10)
        
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension  
    
    #add the most recent polynomial coefficients and radius of curvature
    def add_fit(self, fit, curverad):
        if fit is None:
            print('line: removed fit')
            self.recent_fits.popleft()
            if len(self.recent_fits) == 0:
                self.detected = False  
        else:
            #do not allow radius that is too small (<30) or differs from the last fit by a factor of 2
            if self.curverad > 0:
                if (curverad < 30 
                    or curverad < self.curverad//self.curverad_factor
                    or curverad > self.curverad * self.curverad_factor):                              
                    return 
                
            self.recent_fits.append((fit[0], fit[1], fit[2]))  
            self.curverad = curverad
            self.best_fit = np.mean(self.recent_fits, axis=0)  
            self.detected = True              
           