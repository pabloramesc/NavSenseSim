import numpy as np

from common.math import derivate
from noise.model import get_pink_noise, get_white_noise, get_brown_noise, saturate, digitalize

class MEMS_sensor(): # generic class of MEMS sensor to use as footprint

    def __init__(self, params):
        # sensor parameters
        self.sample_rate = params['sample_rate']
        self.sample_period = 1/self.sample_rate
        self.range = params['range']
        self.resolution = params['resolution']
        self.offset = params['offset']
        self.scale_factor = 1 + params['scale']/100
        self.noise_density = params['noise_density']
        self.random_walk = params['random_walk']
        self.bias_instability = params['bias_instability']
        self.correlation_time = params['correlation_time']
        # sensor variables
        self.state = None
        self.time_prev = None
        self.brown_noise_prev = None
        self.pink_noise_prev = None

    def initialize(self, value, time_now):
        self.state = [np.float64(value)]
        self.time_prev = time_now
        self.brown_noise_prev = get_brown_noise(self.random_walk, self.sample_period, dt=self.sample_period, brown_noise_prev=0.0)
        self.pink_noise_prev = get_pink_noise(self.bias_instability, self.correlation_time, self.sample_period, dt=self.sample_period, pink_noise_prev=0.0)

    def update(self, value, time_now):
        dt = time_now - self.time_prev
        # compute accelerometer model (add noise, bias, offset, scale deviation and make it digital)
        white_noise = get_white_noise(self.noise_density, self.sample_period, dt)
        brown_noise = get_brown_noise(self.random_walk, self.sample_period, dt, brown_noise_prev=self.brown_noise_prev)
        pink_noise = get_pink_noise(self.bias_instability, self.correlation_time, self.sample_period, dt, pink_noise_prev=self.pink_noise_prev)
        value = value * self.scale_factor + white_noise + brown_noise + pink_noise + self.offset
        value = saturate(value, -self.range, self.range)
        value = digitalize(value, self.range, self.resolution)
        # store variables needed by the next iteration
        self.state.append(np.float64(value))
        self.time_prev = time_now
        self.brown_noise_prev = brown_noise
        self.pink_noise_prev = pink_noise
        return np.float64(value)

    def simulate(self, value, time):
        # check if input data is in correct format
        if type(value) != np.ndarray or type(time) != np.ndarray: raise ValueError('input data vectors must be numpy.ndarray')
        if value.ndim != 1 or value.ndim != 1: raise ValueError('input data vectors must be 1 dimensional')
        if value.size == time.size: N = time.size
        # compute noise model
        dt = np.gradient(time)
        white_noise = get_white_noise(self.noise_density, self.sample_period, dt, nlen=N)
        brown_noise = get_brown_noise(self.random_walk, self.sample_period, dt, nlen=N)
        pink_noise = get_pink_noise(self.bias_instability, self.correlation_time, self.sample_period, dt, nlen=N)
        value = value * self.scale_factor + white_noise + brown_noise + pink_noise + self.offset
        value = saturate(value, -self.range, self.range)
        value = digitalize(value, self.range, self.resolution)
        # store and output simulated data
        self.state = value
        return value



class accelerometer(MEMS_sensor):

    def __init__(self, params):
        super().__init__(params)



class gyroscope(MEMS_sensor): # gyroscope doesnt have brown noise

    def __init__(self, params):
        # sensor parameters
        self.sample_rate = params['sample_rate']
        self.sample_period = 1/self.sample_rate
        self.range = params['range']
        self.resolution = params['resolution']
        self.offset = params['offset']
        self.scale_factor = 1 + params['scale']/100
        self.noise_density = params['noise_density']
        self.bias_instability = params['bias_instability']
        self.correlation_time = params['correlation_time']
        # sensor variables
        self.state = None
        self.time_prev = None
        self.pink_noise_prev = None

    def initialize(self, value, time_now):
        self.state = [np.float64(value)]
        self.time_prev = time_now
        self.pink_noise_prev = get_pink_noise(self.bias_instability, self.correlation_time, self.sample_period, dt=self.sample_period, pink_noise_prev=0.0)

    def update(self, value, time_now):
        dt = time_now - self.time_prev
        # compute accelerometer model (add noise, bias, offset, scale deviation and make it digital)
        white_noise = get_white_noise(self.noise_density, self.sample_period, dt)
        pink_noise = get_pink_noise(self.bias_instability, self.correlation_time, self.sample_period, dt, pink_noise_prev=self.pink_noise_prev)
        value = value * self.scale_factor + white_noise + pink_noise + self.offset
        value = saturate(value, -self.range, self.range)
        value = digitalize(value, self.range, self.resolution)
        # store variables needed by the next iteration
        self.state.append(np.float64(value))
        self.time_prev = time_now
        self.pink_noise_prev = pink_noise
        return np.float64(value)

    def simulate(self, value, time):
        # check if input data is in correct format
        if type(value) != np.ndarray or type(time) != np.ndarray: raise ValueError('input data vectors must be numpy.ndarray')
        if value.ndim != 1 or value.ndim != 1: raise ValueError('input data vectors must be 1 dimensional')
        if value.size == time.size: N = time.size
        # compute noise model
        dt = np.gradient(time)
        white_noise = get_white_noise(self.noise_density, self.sample_period, dt, nlen=N)
        pink_noise = get_pink_noise(self.bias_instability, self.correlation_time, self.sample_period, dt, nlen=N)
        value = value * self.scale_factor + white_noise + pink_noise + self.offset
        value = saturate(value, -self.range, self.range)
        value = digitalize(value, self.range, self.resolution)
        # store and output simulated data
        self.state = value
        return value



class magnetometer(MEMS_sensor):

    def __init__(self, params):
        super().__init__(params)



class IMU():

    def __init__(self, model='acc-gyr-mag', acc_params=None, gyr_params=None, mag_params=None):
        """
        An IMU simulation model.

        Parameters
        ----------
        model : str, default: 'acc-gyr-mag'
            Model to choose between 'acc-gyr', 'acc-mag' or 'acc-gyr-mag'.
        acc_params : dict, default: None
            Dicitonary with accelerometer model parameters.
        gyr_params : dict, default: None
            Dicitonary with gyroscope model parameters.
        mag_params : dict, default: None
            Dicitonary with magnetometer model parameters.

        Attributes
        ----------
        n_accels : int
            Number of accelerometers inside the IMU (0 or 3).
        n_gyros : int
            Number of gyroscopes inside the IMU (0 or 3).
        n_magnets : int
            Number of magnetometers inside the IMU (0 or 3).
        accels : list
            List of accelerometers objects.
        gyros : list
            List of gyroscopes objects.
        magnets : list
            List of magnetometers objects.
        
        """
        self.model = model
        # check selected model and store the numbers of each sensor to create
        if model == 'acc-gyr': self.n_accels = 3; self.n_gyros = 3; self.n_magnets = 0
        elif model == 'acc-mag': self.n_accels = 3; self.n_gyros = 0; self.n_magnets = 3
        elif model == 'acc-gyr-mag': self.n_accels = 3; self.n_gyros = 3; self.n_magnets = 3
        else: raise ValueError('Not valid IMU model. Valid models are "acc-gyr", "acc-mag" or "acc-gyr-mag" but gets {}'.format(self.model))
        # build the sensors
        self.accels = []; self.gyros = []; self.magnets = []
        for k in range(self.n_accels): self.accels.append(accelerometer(acc_params))
        for k in range(self.n_gyros): self.gyros.append(gyroscope(gyr_params))
        for k in range(self.n_magnets): self.magnets.append(magnetometer(mag_params))
    
    def initialize(self, acc = None, gyr = None, mag = None, t = 0.0):
        """
        Initializate IMU model with 1 sample.

        Parameters
        ----------
        acc : np.ndarray, default: None
            1-by-3 array with 1 sample of accelerometer real data.
        gyr : np.ndarray, default: None
            1-by-3 array with 1 sample of gyroscope real data.
        mag : np.ndarray, default: None
            1-by-3 array with 1 sample of magnetometer real data.
        t : float, default: 0.0
            Simulation initial time to initializate internal clock.

        Attributes
        ----------
        
        """
        for k in range(self.n_accels):
            self.accels[k].initialize(acc[k], t)
        for k in range(self.n_gyros):
            self.gyros[k].initialize(gyr[k], t)
        for k in range(self.n_magnets):
            self.magnets[k].initialize(mag[k], t)

    def update(self, acc = None, gyr = None, mag = None, t = None):
        """
        Update one step with 1 sample of the IMU simulation state.

        Parameters
        ----------
        acc : np.ndarray, default: None
            1-by-3 array with 1 sample of accelerometer real data.
        gyr : np.ndarray, default: None
            1-by-3 array with 1 sample of gyroscope real data.
        mag : np.ndarray, default: None
            1-by-3 array with 1 sample of magnetometer real data.
        t : float, default: None
            Simulation initial time to initializate internal clock.

        Attributes
        ----------
        accels_out : list
            List of computed accelerometer outputs.
        gyros_out : list
            List of computed gyroscope outputs.
        magnets_out : list
            List of computed magnetometer outputs.
        
        Returns
        ----------
        A tuple of 2 or 3 3-by-N arrays with simulated data of each sensor.
        For example, if model chooesen is 'acc-gyr' you will get (acc_data, gyr_data)
        with acc_data as 3-by-N np.ndarray.
        
        """
        accels_out = []
        for k in range(self.n_accels):
            self.accels[k].update(acc[k], t)
        gyros_out = []
        for k in range(self.n_gyros):
            self.gyros[k].update(gyr[k], t)
        magnets_out = []
        for k in range(self.n_magnets):
            self.magnets[k].update(mag[k], t)

        if self.model == 'acc-gyr':
            return (np.array(accels_out, dtype=np.float64), np.array(gyros_out, dtype=np.float64))
        elif self.model == 'acc-mag':
            return (np.array(accels_out, dtype=np.float64), np.array(magnets_out, dtype=np.float64))
        elif self.model == 'acc-gyr-mag':
            return (np.array(accels_out, dtype=np.float64), np.array(gyros_out, dtype=np.float64), np.array(magnets_out, dtype=np.float64))
        else:
            raise ValueError('Not valid IMU model. Valid models are "acc-gyr", "acc-mag" or "acc-gyr-mag" but gets {}'.format(self.model))

    def simulate(self, acc = None, gyr = None, mag = None, t = None):
        """
        Compute all simulation by adding N steps of N samples to
        the IMU simulation state.
        
        Can be used to compute all simulation data or to append
        a block of data instead of using a loop of .update().

        Parameters
        ----------
        acc : np.ndarray, default: None
            1-by-3 array with 1 sample of accelerometer real data.
        gyr : np.ndarray, default: None
            1-by-3 array with 1 sample of gyroscope real data.
        mag : np.ndarray, default: None
            1-by-3 array with 1 sample of magnetometer real data.
        t : float, default: None
            Simulation initial time to initializate internal clock.

        Attributes
        ----------
        accels_out : list
            List of computed accelerometer outputs.
        gyros_out : list
            List of computed gyroscope outputs.
        magnets_out : list
            List of computed magnetometer outputs.
        
        Returns
        ----------
        A tuple of 2 or 3 3-by-N arrays with simulated data of each sensor.
        For example, if model chooesen is 'acc-gyr' you will get (acc_data, gyr_data)
        with acc_data as 3-by-N np.ndarray.
        
        """
        accels_out = []
        for k in range(self.n_accels):
            accels_out.append(self.accels[k].simulate(acc[k], t))
        gyros_out = []
        for k in range(self.n_gyros):
            gyros_out.append(self.gyros[k].simulate(gyr[k], t))
        magnets_out = []
        for k in range(self.n_magnets):
            magnets_out(self.magnets[k].simulate(mag[k], t))

        if self.model == 'acc-gyr':
            return (np.array(accels_out), np.array(gyros_out))
        elif self.model == 'acc-mag':
            return (np.array(accels_out), np.array(magnets_out))
        elif self.model == 'acc-gyr-mag':
            return (np.array(accels_out), np.array(gyros_out), np.array(magnets_out))
        else:
            raise ValueError('Not valid IMU model. Valid models are "acc-gyr", "acc-mag" or "acc-gyr-mag" but gets {}'.format(self.model))

    def history(self):
        """
        Return all stored data inside the sensor objects. Each
        object will return its state array with all the simulated
        outputs. Dependeding on the choosen model, you will get a
        tuple of 2 or 3 N-by-3 arrays.
        
        Returns
        ----------
        A tuple of 2 or 3 3-by-N arrays with simulated data of each sensor.
        For example, if model chooesen is 'acc-gyr' you will get (acc_data, gyr_data)
        with acc_data as 3-by-N np.ndarray.
        
        """
        accels_out = []
        for k in range(self.n_accels):
            accels_out.append(self.accels[k].state)
        gyros_out = []
        for k in range(self.n_gyros):
            gyros_out.append(self.gyros[k].state)
        magnets_out = []
        for k in range(self.n_magnets):
            magnets_out.append(self.magnets[k].state)

        if self.model == 'acc-gyr':
            return (np.array(accels_out), np.array(gyros_out))
        elif self.model == 'acc-mag':
            return (np.array(accels_out), np.array(magnets_out))
        elif self.model == 'acc-gyr-mag':
            return (np.array(accels_out), np.array(gyros_out), np.array(magnets_out))
        else:
            raise ValueError('Not valid IMU model. Valid models are "acc-gyr", "acc-mag" or "acc-gyr-mag" but gets {}'.format(self.model))
