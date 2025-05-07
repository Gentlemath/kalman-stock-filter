import numpy as np
from abc import ABC, abstractmethod
from pykalman import KalmanFilter

class BaseFilter(ABC):
    @abstractmethod
    def initialize(self, initial_data: np.array):
        pass

    @abstractmethod
    def update(self, new_observation: float) -> float:
        pass

    @abstractmethod
    def run(self, new_observations):
        pass

class BayesianFilter(BaseFilter):
    def __init__(self, window_size = 5, processs_variance = 0.1):
        self.window_size = window_size
        self.process_variance = processs_variance
        self.prices = None
        self.current_mean = None
        self.corrent_var = None

    def initialize(self, initial_data):
        self.prices = [float(x) for x in initial_data[-self.window_size:]]
        self.current_mean = np.mean(self.prices)
        self.current_var = np.var(self.prices)

    def update(self, observation):
        self.prices.append(float(observation))
        if len(self.prices) > self.window_size:
            self.prices.pop(0)
        try:
            likelihood_mean = np.mean(self.prices)
            likelihood_var = np.var(self.prices)
        except (TypeError, ValueError):
            breakpoint()
            print(self.prices)
            raise ValueError("Something wrong here!")

        self.current_mean = (likelihood_var * self.current_mean + 
                             self.current_var*likelihood_mean)/(likelihood_var + self.current_var)    
        self.current_var = (likelihood_var * self.current_var)/(likelihood_var + self.current_var)
        self.current_var += self.process_variance

        return self.current_mean


    def run(self, observations):
        ests = []
        for obs in observations:
            self.update(obs)
            ests.append(self.current_mean)
        
        return ests
    

class KalmanFilterImpl(BaseFilter):
    def __init__(self, transition_matrices = 1, observation_matrices = 1,
                 initial_state_mean = 0, initial_state_covariance = 1,
                 observation_covariance = 1, transition_covariance = 0.1):
        self.kf = KalmanFilter(
            transition_matrices = transition_matrices,
            observation_matrices = observation_matrices,
            initial_state_mean = initial_state_mean,
            initial_state_covariance = initial_state_covariance,
            observation_covariance = observation_covariance,
            transition_covariance = transition_covariance
        )
        self.state_mean = None
        self.state_cov = None

    def initialize(self, initial_data):
        self.state_mean = float(initial_data[-1])  # Use last observed price
        self.state_cov = self.kf.initial_state_covariance

    def update(self, new_observation):
        self.state_mean, self.state_cov = self.kf.filter_update(self.state_mean, self.state_cov, new_observation)

    def run(self, observations):
        ests = []
        for obs in observations:
            self.update(obs)
            val = float(np.ma.getdata(self.state_mean).item())
            ests.append(val)
        return ests
    
class ExtendedKalman(BaseFilter):   ## no nonlinear here, actually a kalman filter
    def __init__(self, process_noise = 0.1,  measurement_noise = 1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = None
        self.covariance = None
    def initialize(self, initial_data):
        self.state = initial_data[-1]  # last observed
        self.covariance = 1.0
    
    def update(self, new_observation):
        predict_state = self.state # random walk
        predict_covariance = self.covariance + self.process_noise

        innovation = new_observation - predict_state
        innovation_covariance = predict_covariance + self.measurement_noise
        kalman_gain = predict_covariance/innovation_covariance

        self.state = self.state + kalman_gain * innovation
        self.covariance = (1 - kalman_gain) * predict_covariance

    def run(self, new_observations):
        ests = []
        for obs in new_observations:
            self.update(obs)
            ests.append(self.state)
        return ests

        











        


