---
title: 'Hierarchical `PreProFit`: multi-object, multi-instrument, multi-parameterization fitting of cluster pressure profiles and population properties'
tags:
  - Python
  - 
authors:
  - name: Fabio Castagna
    orcid: 0000-0003-1324-9641
    affiliation: "1"  
  - name: Stefano Andreon
    affiliation: "1"  

affiliations:
 - name: INAF-Osservatorio Astronomico di Brera, via Brera 28, 20121 Milano, Italy

   index: 1

date: xx March 2026
bibliography: paper.bib



# Summary

We present an updated and expanded version of `PreProFit`, a flexible, publicly available software pipeline designed for the analysis of galaxy cluster pressure profiles from Sunyaev-Zeldovich (SZ) data. While the original release was limited to individual cluster modeling, hierarchical `PreProFit` introduces a hierarchical Bayesian framework that enables the simultaneous inference of individual profiles and global population quantities. This allows users to characterize the population-averaged pressure profile and intrinsic scatter across clusters in a single, self-consistent procedure, also providing more accurate individual estimates by leveraging the population-level posteriors as informative priors.
The pipeline has been redesigned for high performance and modularity, migrating to the `PyMC` library for MCMC estimation and optimizing instrument-specific filtering operations. Hierarchical `PreProFit` supports different parameterizations for the pressure profile, including a new restricted cubic spline model that averts the limitations of previous models. 

# Statement of Need

The Sunyaev-Zeldovich (SZ) effect - a spectral distortion of the Cosmic Microwave Background - serves as a direct tracer of gas pressure within the intracluster medium (ICM). Because the SZ signal depends solely on the pressure integrated along the line of sight, it is a direct tool for probing the spatial distribution of the ICM across galaxy clusters. 
Over the last decade, the increasing availability of high-resolution SZ instruments has shifted the focus from individual objects to population-level studies. 
Estimating the average pressure profile and its associated dispersion across a cluster population offers insights that extend far beyond single-cluster analyses. Specifically, characterizing the profile scatter is essential for quantifying the impact of non-gravitational processes - such as gas cooling, supernova feedback, and Active Galactic Nuclei activity - as well as hydrodynamical behaviors driven by shocks, turbulence, and bulk motions. The pressure profile can also be used to address more complex questions, such as those related to degenerate higher-order scalar-tensor theories. Ultimately, these population studies refine our understanding of cosmic evolution and large-scale structure. 

In \citet{Castagna2019}, we introduced `PreProFit` as the first publicly available code designed to fit galaxy cluster pressure profiles using SZ data. However, that initial version was limited to the analysis of individual clusters with just one possible parametrization - a constraint shared by later tools (e.g., `panco2` \cite{keruzore2023}).
Hierarchical `PreProFit` represents, to the best of our knowledge, the first publicly available pipeline that enables population-level analysis. It allows users to perform a simultaneous inference of both individual cluster profiles and global population parameters, specifically the population-averaged pressure profile and the intrinsic scatter. 
Furthermore, a population-level approach is particularly valuable when individual cluster parameters are poorly constrained; by leveraging information from the entire sample as a self-consistent prior, the hierarchical framework improves both the accuracy and precision of these estimates \cite{Castagna2025}.

Hierarchical `PreProFit` offers unprecedented flexibility across several configurations. It is able to flexibly handle several different configurations in an unprecedented way. While optimized for multi-cluster ensembles, the pipeline retains full support for single-cluster analysis. It seamlessly handles data from various telescopes, automatically processing them according to instrument-specific requirements. Furthermore, users can now choose from several pressure profile parameterizations.

# Software design

Hierarchical `PreProFit` provides several methods for parameterizing the pressure profile of galaxy clusters. The first is the widely adopted generalized Navarro, Frenk \& White (gNFW) profile \citep{Nagai2007}, which offers limited flexibility and features parameters that are often difficult to interpret.
A second option is the piecewise power-law model proposed by \citet{Romero2018}. This is a nonparametric pressure profile that assumes a power-law interpolation between radial bins and is analytically integrable under the assumption of ellipsoidal symmetry; however, a significant drawback of this method is that the profile uncertainties tend to peak at the radii corresponding to the knots.
Another available option is a cubic spline interpolation across different knots. While this provides high flexibility, it can lead to undesired twists at the extremities of the radial range. As an optimal solution, we introduced in \citet{Castagna2025} the restricted cubic spline model \cite{durrleman1989}. This parameterization is highly flexible, combining the elasticity of cubic interpolation between the inner and outer knots with linear regression at both extremities. This avoids unconstrained oscillations at radii where data is sparse-specifically, well inside the beam FWHM and at large radii where the S/N vanishes.
Users can freely specify the number and location of the knots based on their analysis requirements.
To ensure a finite integral and distinguish between a radially flat background and a cluster signal, users can set an upper limit for the slope value beyond a given outer radius, a practice consistent with several works in the literature \citep{Romero2018, Andreon2021}.
Hierarchical `PreProFit` automatically initializes the MCMC starting points for each parameter using values from the universal pressure profile \cite{Arnaud2010}; this ensures rapid and optimal convergence towards the posterior distribution.

As in the previous version, hierarchical `PreProFit` provides a highly flexible, automated method to handle filtering involving the point spread function (PSF) and the transfer function. This method is compatible with different instruments. Users can choose to read these functions from raw data or use Gaussian approximations. In the former case, the user specifies whether the file includes both the beam and the transfer function or if they are provided separately. By simply specifying the provenance of the transfer function, `PreProFit` automatically applies the corresponding filtering procedure during the fitting process.
Compared to the previous version, we have optimized these operations by unifying the beam and transfer function convolutions into a single operation and by shifting all redundant calculations to a pre-computation stage.

Hierarchical `PreProFit` relies on [`PyMC`](https://www.pymc.io/welcome.html)~\citep{Patil2010} to perform MCMC estimations, moving away from the `emcee` package used in the previous version. Thanks to its intuitive interface, `PyMC` is particularly effective for modeling complex dependencies in hierarchical Bayesian frameworks. It supports multidimensional parameters by allowing group-level distributions to define structured dependencies across multiple levels. Furthermore, `PyMC` enables vectorized priors through the “shape” argument, ensuring computational efficiency and the automatic sharing of statistical strength across groups.

Users can easily customize their models to fit specific needs; for instance, prior distributions can be modified using standard `PyMC` syntax. When analyzing a population of galaxy clusters, users may include parameters that account for dependencies on mass or redshift, using either a single global parameter or radial-specific parameters.

The GitHub repository includes two comprehensive examples of these features:
a single-cluster analysis based on NIKA data using a gNFW model, and a multi-object (population) analysis based on SPT data using the restricted cubic spline model. 
To ensure ease of use and reproducibility, we utilize YAML configuration files for each case. These files centralize all necessary parameters allowing users to easily adapt the pipeline to their own datasets.

# Acknowledgements

We acknowledge INAF grant “Characterizing the newly discovered clusters of low surface brightness” and PRIN-MIUR grant 20228B938N “Mass and selection biases of galaxy clusters: a multi-probe approach”, the latter funded by the European Union Next generation EU, Mission 4 Component 1 CUP C53D2300092 0006.

# References
