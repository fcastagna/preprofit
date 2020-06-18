# PreProFit
## Pressure Profile Fitter for galaxy clusters in Python
*Castagna Fabio, Andreon Stefano, Pranjal RS.*

`preprofit` is a Python program that allows to fit the pressure profile of galaxy clusters using MCMC.

`preprofit` is embedded in [`JoXSZ`](https://github.com/fcastagna/JoXSZ), our complete program that allows to jointly fit the thermodynamic profiles of galaxy clusters from both SZ and X-ray data.

As an example, we show the application of `preprofit` on the high-redshift cluster of galaxies CL J1226.9+3332 (z = 0.89).
Beam data and transfer function data come from the [NIKA data release](http://lpsc.in2p3.fr/NIKA2LPSZ/nika2sz.release.php).

PLEASE NOTE that the transfer function filtering method has been changed on March 30th, 2020.

### Requirements
`preprofit` requires the following:
- [mbproj2](https://github.com/jeremysanders/mbproj2)
- [PyAbel](https://github.com/PyAbel/PyAbel)
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [astropy](http://www.astropy.org/)
- [emcee](https://emcee.readthedocs.io/)
- [six](https://pypi.org/project/six/)
- [matplotlib](https://matplotlib.org/)
- [corner](https://pypi.org/project/corner/)

### Credits
Castagna Fabio, Andreon Stefano, Pranjal RS.

For more details, see [Castagna and Andreon, A&A, 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..22C/abstract).
