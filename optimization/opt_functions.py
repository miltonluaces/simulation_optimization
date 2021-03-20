import sys
sys.path.append('D:/source/repos')
from utilities.std_imports import *
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
import scipy.optimize as so
from numpy import abs, sin, cos, exp, mean, pi, prod, sqrt, sum, multiply, power, e

### Show 2D functions

def show_function(func, rg, gr=0.1):
    xs = np.arange(rg[0], rg[1], gr)
    x, y = np.meshgrid(xs, xs)
    
    fig = plt.figure(figsize=[20,10])
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, func([x, y]), cmap='jet');
    
### nD functions

def sphere(x):
    return sum(power(x,2))

def sum2(x):
    n = len(x)
    j = np.arange( 1., n+1 )
    return sum( j * power(x,2))

def ackley(x):
    return -20.0 * exp(-0.2 * sqrt(0.5 * sum(power(x, 2)))) - exp(0.5 * sum(cos(multiply(x,2*pi)))) + e + 20

def dixon(x):  
    n = len(x)
    j = np.arange(2, n+1)
    x2 = 2 * power(x,2)
    return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2

def griewank(x, fr=4000 ):
    n = len(x)
    j = np.arange(1., n+1)
    s = sum(power(x,2))
    p = prod(cos(x/sqrt(j)))
    return s/fr - p + 1

def levy(x):
    n = len(x)
    z = np.add(np.divide(np.add(x, - 1), 4),1)
    return (sin( pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))

def michalewicz(x): 
    n = len(x)
    j = np.arange( 1., n+1 )
    return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * .5) )

def rastrigin(x):  
    n = len(x)
    return 10*n + sum( power(x,2) - 10 * cos( multiply(x, 2*pi)))

def schwefel(x):  
    n = len(x)
    return 418.9829 * n - sum( x * sin( sqrt( abs( x ))))

def nesterov(x):
    x0 = x[:-1]
    x1 = x[1:]
    return abs( 1 - x[0] ) / 4 \
        + sum( abs( x1 - 2*abs(x0) + 1 ))

def schaffer(x):
    return -1

def rosenbrock(x):
    return so.rosen(x)

### 2D functions

def sphere2(x):
    x1, x2 = x
    return x1**2 + x2**2

def sum22(x):
    x1, x2 = x
    return 1 * x1**2 + 2 * x2**2

def ackley2(x):
    x1, x2 = x
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x1**2 + x2**2))) - exp(0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2))) + e + 20

def dixon2(x):
    return(dixon(x))

def griewank2(x, fr=4000):
    s = x[0]**2 + x[1]**2
    p = cos(x[0]/sqrt(1)) * cos(x[1]/sqrt(2))
    return s/fr - p + 1

def levy2(x):
    return levy(x)

def michalewicz2(x): 
    x1, x2 = x
    return - ( sin(x1) * sin( 1 * x1**2 / pi ) ** (2 * .5) + sin(x2) * sin( 2 * x2**2 / pi ) ** (2 * .5)) 

def rastrigin2(x): 
    x1, x2 = x
    return 10*2 + ( x1**2 - 10 * cos( 2 * pi * x1)) +( x2**2 - 10 * cos( 2 * pi * x2))

def schwefel2(x):  
    x1, x2 = x
    return 418.9829 * 2 - x1 * sin( sqrt( abs( x1 ))) - x2 * sin( sqrt( abs( x2 )))

def nesterov2(x):
    return nesterov(x)
    
def schaffer2(x):
    x1, x2 = x
    return 0.5 + ((sin(x1**2 - x2**2))**2 - 0.5) / (1 + 0.001 * x1**2 + x2**2)**2

def rosenbrock2(x):
    return so.rosen(x)


## for review

def saddle(x):
    return np.mean( np.diff( x **2 )) + .5 * np.mean( x **4 )

def perm(x, b=.5):
    n = len(x)
    j = np.arange( 1., n+1 )
    xbyj = np.fabs(x) / j
    return mean([ mean( (j**k + b) * (xbyj ** k - 1) ) **2
            for k in j/n ])

def powell(x):
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append( x, np.zeros( n4 - n ))
    x = x.reshape(( 4, -1 ))  
    f = np.empty_like( x )
    f[0] = x[0] + 10 * x[1]
    f[1] = sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2]) **2
    f[3] = sqrt(10) * (x[0] - x[3]) **2
    return sum( f**2 )

def ellipse(x):
    return mean( power(np.add(multiply(x,-1), 1),2)) + 100 * mean( power(np.diff(x),2))

def powersum( x, b=[8,18,44,114] ):  # power.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    s = 0
    for k in range( 1, n+1 ):
        bk = b[ min( k - 1, len(b) - 1 )]  # ?
        s += (sum( x**k ) - bk) **2  # dim 10 huge, 100 overflows
    return s

def trid( x ):
    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )

def zakharov( x ):  
    n = len(x)
    j = np.arange( 1., n+1 )
    s2 = sum( j * x ) / 2
    return sum( x**2 ) + s2**2 + s2**4