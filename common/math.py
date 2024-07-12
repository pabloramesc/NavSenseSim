import numpy as np

def derivate(f, t=None):
    """
    La derivación se realiza por 'diferencias hacia adelante':
        f'(n) = (f(n+1) - f(n))/(t(n+1) - t(n)) / n = 0,1,2 ... N
    Esto es debido a que es la forma de derivar que emplean la mayoría de algoritmos en tiempo real.
    El primer valor de la derivada debe ser nulo: f'(1) = 0
    Si no se proporciona la variable independiente t de f(t) se derivará como:
        f'(n) = f(n+1) - f(n) / n = 0,1,2 ... N
    """
    if not np.array(t).any():
        return np.append(0, np.diff(f))
    else:        
        if len(f) != len(t): raise ValueError('f and t must be same length')
        return np.append(0, np.diff(f)/np.diff(t))

def integrate(f, f0=0, t=None):
    """
    La integración se realiza por 'regla del rectángulo' o 'suma acumulada':
        F(n) = F(n-1) + f(n)*(t(n+1) - t(n)) / n = 1,2,3 ... N
    Debe proporcionarse el valor inicial de la función integrada como f0 = F(0) o se usará 0 por defecto.
    Si no se proporciona la variable independiente t de f(t) se integrará como:
        F(n) = F(n-1) + f(n) / n = 1,2,3 ... N
    """
    if not np.array(t).any():
        return f0 + np.cumsum(f)
    else:
        if len(f) != len(t): raise ValueError('f and t must be same length')
        return f0 + np.cumsum(f*derivate(t))

def rms_error(pred: np.array, real: np.ndarray):
    """
    Compute the Root Mean Square error value for the input data.

    pred : np.ndarray
        N array of predicted data
    real : np.ndarray
        N array of real data
    """
    if pred.shape!= real.shape:
        raise ValueError("pred and real arrays must have same shape. Got {} for pred and {} for real.".format(pred.shape, real.shape))
    N = len(pred)
    rmse = np.sqrt(np.sum((pred-real)**2/N))
    return rmse



if __name__ == "__main__":
    """
    Código para comprobar que la integración y la derivación son reversibles
    """

    import matplotlib.pyplot as plt

    pi = np.pi

    t = np.linspace(-pi,pi,1024)
    f = np.cos(t)

    DF = derivate(f, t)
    IF = integrate(DF, f[0], t)

    plt.figure(1)
    plt.plot(t,f)
    plt.plot(t,DF)
    plt.plot(t,IF)
    plt.legend(('f','DF','IF'))

    plt.figure(2)
    plt.plot(t,f)
    plt.plot(t,DF)
    plt.plot(t,IF)
    plt.legend(('f','DF','IF'))

    plt.show()