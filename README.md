# PreProFit
## Pressure Profile Fitter for galaxy clusters in Python
*Castagna Fabio, Andreon Stefano.*

`preprofit` is a Python program that allows to fit the pressure profile of galaxy clusters using MCMC.

`preprofit` is embedded in [`JoXSZ`](https://github.com/fcastagna/JoXSZ), our complete program that allows to jointly fit the thermodynamic profiles of galaxy clusters from both SZ and X-ray data.

As an example, we show the application of `preprofit` on the high-redshift cluster of galaxies SPT-CLJ0500-5116.

PLEASE NOTE that the transfer function filtering method has been changed on March 30th, 2020.

### Requirements
`preprofit` requires the following:
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [astropy](http://www.astropy.org/)
- [pymc](https://www.pymc.io/)
- [matplotlib](https://matplotlib.org/)
- [corner](https://pypi.org/project/corner/)
- [arviz](https://python.arviz.org/)

### Credits
Castagna Fabio, Andreon Stefano.

For more details, see [Castagna and Andreon, A&A, 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..22C/abstract).
