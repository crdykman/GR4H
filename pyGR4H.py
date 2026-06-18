## Link GR4J Model with SPOTPY

import pandas as pd

#Import Relevant Packages
from GR4H_model import GR4H
from spotpy.objectivefunctions import rmse, nashsutcliffe
from spotpy.parameter import Uniform


#Create Spotpy setup class
class spot_setup():
    
    #Initialise Parameters
    x1  =  Uniform(low=1.0, high=1500.0) # Maximum production capacity (mm)
    x2  =  Uniform(low=-10.0, high=10.0) # Water exchange coefficient, which is positive in case of a gain, and negative in case of a loss, or null (mm)
    x3  =  Uniform(low=1.0, high=1000.0) # Routing maximum capacity (mm)
    x4  =  Uniform(low=0.5, high=500) # Unit hydrograph time base (hours)

    def __init__(self, climatefile, area, obj_func=None, ps0=1.0, rs0=0.5,
                events=None, obj_freq=None):

        #Load Observation data from file
        climatedata = pd.read_csv(climatefile, 
                                  index_col=0, 
                                  parse_dates=True,
                                  dayfirst=True)
        self.forcings = climatedata[['prec', 'pet']] 
        self.trueObs = climatedata['qt'] 
        # catchment size
        self.area = area #km2
        #transformation factor
        self.Factor = self.area / 3.6 #Convert units mm/hr to m3/s
        #objective function
        self.obj_freq = obj_freq
        self.obj_func = obj_func
        #calibration events
        self.events = events
        #Initial Conditions
        self.ps0 = ps0
        self.rs0 = rs0 

    def simulation(self,x):
        #Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        model = GR4H(area=self.area, params={'ps0': self.ps0,  
                                            'rs0': self.rs0,
                                            'x1' :x[0],
                                            'x2' :x[1],
                                            'x3' :x[2],
                                            'x4' :x[3]})
        data = model.run(self.forcings)
        #The first year of simulation data is ignored (warm-up)
        if self.events is None:
            return data.qt.iloc[366*24:]
        return data.qt.iloc[self.events]
    
    def evaluation(self):
        #The first year of simulation data is ignored (warm-up)
        if self.events is None:
            return self.trueObs.iloc[366*24:]
        return self.trueObs.iloc[self.events]

    def objectivefunction(self,simulation,evaluation, params=None):
        #SPOTPY expects to get one or multiple values back, 
        #that define the performance of the model run
        if self.obj_freq == 'D':
            simulation = simulation.resample('D').sum().iloc[1:-1]
            evaluation = evaluation.resample('D').sum().iloc[1:-1]
        
        if self.obj_func is None:
            # This is used if not overwritten by user
            like = -rmse(evaluation,simulation)
        elif self.obj_func == "nnse":
            like =  1/(2-nashsutcliffe(evaluation,simulation))
        elif self.obj_func == "-nnse":
            like =  -1/(2-nashsutcliffe(evaluation,simulation))
        else:
            #Way to ensure flexible spot setup class
            like = self.obj_func(evaluation,simulation)    
        return like