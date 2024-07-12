# %% 
# IMPORT EXPERIMENTAL DATA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get data from file
file_name = r"data\imu\tes0.dat"
input_data = pd.read_csv(
    file_name, sep=",", skipfooter=1, engine="python"
)  # to ignore last row skipfooter needs run on python engine
input_data.columns = ["ax", "ay", "az", "wx", "wy", "wz", "temp", "dt"]

# display data content
input_data

# plot imported data
input_data.plot(
    subplots=True,
    legend=False,
    figsize=(20, 15),
    title=["ax", "ay", "az", "wx", "wy", "wz", "temp", "dt"],
)
plt.show()

# %% 
# SLICE DATA AND REMOVE NOISY READINGS

# extract samples between 2 indices
sliced_data = input_data.iloc[1000000:4500000].reset_index(drop=True)

def is_noisy(data: pd.DataFrame, th: float):
    return (
        (abs(data["ax"] - data["ax"].mean()) > th * data["ax"].std(0))
        | (abs(data["ay"] - data["ay"].mean()) > th * data["ay"].std(0))
        | (abs(data["az"] - data["az"].mean()) > th * data["az"].std(0))
        | (abs(data["wx"] - data["wx"].mean()) > th * data["wx"].std(0))
        | (abs(data["wy"] - data["wy"].mean()) > th * data["wy"].std(0))
        | (abs(data["wz"] - data["wz"].mean()) > th * data["wz"].std(0))
    )

# remove samples that deviate by more than 5 times the standard deviation from the mean value
noisy_samples_indices = np.where(is_noisy(sliced_data, th=5.0))[0]
clean_data = sliced_data.drop(index=noisy_samples_indices)

# plot cleaned data
clean_data.plot(
    subplots=True,
    legend=False,
    figsize=(20, 15),
    title=["ax", "ay", "az", "wx", "wy", "wz", "temp", "dt"],
)
plt.show()

# %% 
# READINGS CONVERSION AND SAMPLE RATE CALCULATION

from common.constants import GRAV

# accelerations (original accel data is in mg)
ax = clean_data["ax"] * 1e-3 * GRAV  # from mg to m/s^2
ay = clean_data["ay"] * 1e-3 * GRAV  # from mg to m/s^2
az = clean_data["az"] * 1e-3 * GRAV  # form mg to m/s^2

a_mag = np.sqrt(ax**2 + ay**2 + az**2)  # compute acceleration magnitude

real_accel = np.array([ax, ay, az])

# angular rates (original gyro data is in deg/s)
wx = clean_data["wx"]
wy = clean_data["wy"]
wz = clean_data["wz"]

real_gyro = np.array([wx, wy, wz])

# temperature (original temp data is in Celsius degrees)
temp = clean_data["temp"]

# time (original dt data is ms)
dt = clean_data["dt"] * 1e-3  # from ms to s
t = np.array(np.cumsum(dt))  # compute time from dt
hours = t / 3600  # save time as hours for plotting

# store modified data as pandas dataframe
mod_data = pd.DataFrame(
    np.transpose([t, ax, ay, az, a_mag, wx, wy, wz, temp, dt]),
    columns=["t", "ax", "ay", "az", "a_mag", "wx", "wy", "wz", "temp", "dt"],
)

# display data content
mod_data

mod_data.describe()

# plot modified data
mod_data.plot(
    x="t",
    subplots=True,
    legend=False,
    figsize=(20, 15),
    title=["ax", "ay", "az", "a_mag", "wx", "wy", "wz", "temp", "dt"],
)
plt.show()

# plot total elapsed time
import datetime

print("Sampling Time: ", datetime.timedelta(seconds=mod_data["t"].max()))

# find a average sampling rate
print(mod_data["dt"].describe())
print()
print("mode =", mod_data["dt"].mode())

# store sampling rate
sample_period = mod_data["dt"].mode()[0]
sample_rate = 1 / sample_period

print("Sample Rate = {:.2f} Hz".format(sample_rate))


# %% 
# CALCULATE THE ALLAN VARIANCE FOR ACCELEROMETER

from noise.allan import compute_allan_variance, get_allan_N, get_allan_B, get_allan_K

max_clusters_num = 100

accel_adevs = []
accel_Ns, accel_Bs, accel_Ks = [], [], []
accel_Tcs = []
for acc in real_accel:
    # compute allan variance and allan standard deviation
    avar, tau = compute_allan_variance(acc, sample_rate, max_clusters_num)
    adev = np.sqrt(avar)
    # compute N, B and K noise factors (Tc is the correlation time for B)
    N = get_allan_N(tau, adev)
    B, Tc = get_allan_B(tau, adev)
    K = get_allan_K(tau, adev)
    # store allan standard deviation and noise factors
    accel_adevs.append(adev)
    accel_Ns.append(N)
    accel_Bs.append(B)
    accel_Ks.append(K)
    accel_Tcs.append(Tc)

accel_N = max(accel_Ns)
accel_B = max(accel_Bs)
accel_K = max(accel_Ks)
accel_Tc = accel_Tcs[accel_Bs.index(accel_B)]  # get the correlation time for selected B

plt.figure(1, figsize=(10, 5))
plt.title("Allan Standard Deviation of Accelerometers")
plt.loglog(tau, accel_adevs[0])
plt.loglog(tau, accel_adevs[1])
plt.loglog(tau, accel_adevs[2])
plt.loglog(tau, accel_N / np.sqrt(tau), "r--")
plt.loglog(1, accel_N, "ro")
plt.loglog(tau, accel_B * np.ones(tau.size), "y--")
plt.loglog(accel_Tc, accel_B, "yo")
plt.loglog(tau, accel_K * np.sqrt(tau / 3), "b--")
plt.loglog(3, accel_K, "bo")
plt.xlabel("tau (s)")
plt.ylabel("ADEV (m/s^2)")
plt.grid(which="both", linestyle=":")

print("N = {:.4e} m/s^2/sqrt(Hz)".format(accel_N))
print("B = {:.4e} m/s^2".format(accel_B))
print("K = {:.4e} m/s^3*sqrt(Hz)".format(accel_K))
print("Tc = {:.2} s".format(accel_Tc))


# %% 
# CALCULATE THE ALLAN VARIANCE FOR GYROSCOPE

gyro_adevs = []
gyro_Ns, gyro_Bs = [], []
gyro_Tcs = []
for gyr in real_gyro:
    # compute allan variance and allan standard deviation
    avar, tau = compute_allan_variance(gyr, sample_rate, max_clusters_num)
    adev = np.sqrt(avar)
    # compute N and K noise factors
    N = get_allan_N(tau, adev)
    B, Tc = get_allan_B(tau, adev)
    # store allan standard deviation and noise factors
    gyro_adevs.append(adev)
    gyro_Ns.append(N)
    gyro_Bs.append(B)
    gyro_Tcs.append(Tc)

gyro_N = max(gyro_Ns)
gyro_B = max(gyro_Bs)
gyro_Tc = gyro_Tcs[gyro_Bs.index(gyro_B)]  # get the correlation time for selected B

plt.figure(2, figsize=(10, 5))
plt.title("Allan Standard Deviation of Gyroscopes")
plt.loglog(tau, gyro_adevs[0])
plt.loglog(tau, gyro_adevs[1])
plt.loglog(tau, gyro_adevs[2])
plt.loglog(tau, gyro_N / np.sqrt(tau), "r--")
plt.loglog(1, gyro_N, "ro")
plt.loglog(tau, gyro_B * np.ones(tau.size), "y--")
plt.loglog(gyro_Tc, gyro_B, "yo")

plt.xlabel("tau (s)")
plt.ylabel("ADEV (deg/s)")
plt.grid(which="both", linestyle=":")

print("N = {:.4e} deg/s/sqrt(Hz)".format(gyro_N))
print("B = {:.4e} deg/s".format(gyro_B))
print("Tc = {:.2} s".format(gyro_Tc))


# %% 
# RUN IMU SIMULATION WITH CHARACTERIZED PARAMETERS

from sensors.imu import IMU

# accel and gyro params stored as dictionary
accel_params = {
    "sample_rate": sample_rate,  # 0 - sample rate (Hz)
    "range": 2 * GRAV,  # 1 - range (m/s^2)
    "resolution": 16,  # 2 - ADC resolution (bits)
    "offset": 0,  # 3 - offset (m/s^2)
    "scale": 0,  # 4 - scale +- deviation(%)
    "noise_density": accel_N,  # 5 - noise density (m/s^2/sqrt(Hz))
    "random_walk": accel_K,  # 6 - random walk (m/s^3*sqrt(Hz))
    "bias_instability": accel_B,  # 7 - bias instability (m/s^2)
    "correlation_time": accel_Tc,  # 8 - correlation time (s)
}

gyro_params = {
    "sample_rate": sample_rate,  # 0 - sample rate (Hz)
    "range": 250,  # 1 - range (deg/s)
    "resolution": 16,  # 2 - ADC resolution (bits)
    "offset": 0,  # 3 - offset (deg/s)
    "scale": 0,  # 4 - scale +- deviation (%)
    "noise_density": gyro_N,  # 5 - noise density (deg/s/sqrt(Hz))
    "bias_instability": gyro_B,  # 6 - bias instability (deg/s)
    "correlation_time": gyro_Tc,  # 7 - correlation time (s)}
}

# buil imu object
imu = IMU(model="acc-gyr", acc_params=accel_params, gyr_params=gyro_params)

# compute ideal data as constant value as average of taken data
ideal_accel = np.array(
    [
        np.average(real_accel[0]) * np.ones(real_accel[0].shape),
        np.average(real_accel[1]) * np.ones(real_accel[1].shape),
        np.average(real_accel[2]) * np.ones(real_accel[2].shape),
    ]
)

ideal_gyro = np.array(
    [
        np.average(real_gyro[0]) * np.ones(real_gyro[0].shape),
        np.average(real_gyro[1]) * np.ones(real_gyro[1].shape),
        np.average(real_gyro[2]) * np.ones(real_gyro[2].shape),
    ]
)

# simulate sensors
imu.initialize(acc=ideal_accel[:, 0], gyr=ideal_gyro[:, 0], t=t[0])
imu.simulate(acc=ideal_accel, gyr=ideal_gyro, t=t)
sim_accel, sim_gyro = imu.history()

# plot simulated data vs ideal data vs real data
plt.figure(1, figsize=(15, 5))
plt.subplot(331)
plt.plot(hours, real_accel[0])
plt.title("Real accel")
plt.ylabel("Accel X (m/s^2)")
plt.subplot(332)
plt.plot(hours, ideal_accel[0])
plt.title("Ideal accel")
plt.subplot(333)
plt.plot(hours, sim_accel[0])
plt.title("Simulated accel")
plt.subplot(334)
plt.plot(hours, real_accel[1])
plt.ylabel("Accel Y (m/s^2)")
plt.subplot(335)
plt.plot(hours, ideal_accel[1])
plt.subplot(336)
plt.plot(hours, sim_accel[1])
plt.subplot(337)
plt.plot(hours, real_accel[2])
plt.xlabel("Time (hours)")
plt.ylabel("Accel Z (m/s^2)")
plt.subplot(338)
plt.plot(hours, ideal_accel[2])
plt.xlabel("Time (hours)")
plt.subplot(339)
plt.plot(hours, sim_accel[2])
plt.xlabel("Time (hours)")

plt.figure(2, figsize=(15, 5))
plt.subplot(331)
plt.plot(hours, real_gyro[0])
plt.title("Real gyro")
plt.ylabel("Gyro X (deg/s)")
plt.subplot(332)
plt.plot(hours, ideal_gyro[0])
plt.title("Ideal gyro")
plt.subplot(333)
plt.plot(hours, sim_gyro[0])
plt.title("Simulated gyro")
plt.subplot(334)
plt.plot(hours, real_gyro[1])
plt.ylabel("Gyro Y (deg/s)")
plt.subplot(335)
plt.plot(hours, ideal_gyro[1])
plt.subplot(336)
plt.plot(hours, sim_gyro[1])
plt.subplot(337)
plt.plot(hours, real_gyro[2])
plt.xlabel("Time (hours)")
plt.ylabel("Gyro Z (deg/s)")
plt.subplot(338)
plt.plot(hours, ideal_gyro[2])
plt.xlabel("Time (hours)")
plt.subplot(339)
plt.plot(hours, sim_gyro[2])
plt.xlabel("Time (hours)")

plt.show()


# %% 
# COMPUTE ALLAN VARIANCE FOR THE SIMULATION

max_clusters_num = 100

# compute accel adev
sim_accel_adevs = []
for acc in sim_accel:
    avar, tau = compute_allan_variance(acc, sample_rate, max_clusters_num)
    adev = np.sqrt(avar)
    sim_accel_adevs.append(adev)

# compute gyro adev
sim_gyro_adevs = []
for gyr in sim_gyro:
    avar, tau = compute_allan_variance(gyr, sample_rate, max_clusters_num)
    adev = np.sqrt(avar)
    sim_gyro_adevs.append(adev)

# plot allan variances

plt.figure(1, figsize=(10, 5))
plt.title("ADEV for Real Accelerometer")
plt.grid(which="both", linestyle=":")
plt.loglog(tau, accel_adevs[0])
plt.loglog(tau, accel_adevs[1])
plt.loglog(tau, accel_adevs[2])
plt.loglog(tau, sim_accel_adevs[0])
plt.loglog(tau, sim_accel_adevs[1])
plt.loglog(tau, sim_accel_adevs[2])
plt.legend(("real ax", "real ay", "real az", "sim ax", "sim ay", "sim az"), loc=3)
plt.xlabel("tau (s)")
plt.ylabel("ADEV (m/s^2)")

plt.figure(2, figsize=(10, 5))
plt.title("ADEV for Real Gyroscope")
plt.grid(which="both", linestyle=":")
plt.loglog(tau, gyro_adevs[0])
plt.loglog(tau, gyro_adevs[1])
plt.loglog(tau, gyro_adevs[2])
plt.loglog(tau, sim_gyro_adevs[0])
plt.loglog(tau, sim_gyro_adevs[1])
plt.loglog(tau, sim_gyro_adevs[2])
plt.legend(("real wx", "real wy", "real wz", "sim wx", "sim wy", "sim wz"), loc=3)
plt.xlabel("tau (s)")
plt.ylabel("ADEV (deg/s)")

plt.show()


# %% 
# END OF SCRIPT
# press Run Above to execute all the script
print("End of script!")
