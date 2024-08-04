![Tungsten Sample Render](https://raw.githubusercontent.com/daseyb/gpis-light-transport/main/Header.jpg "Tungsten Sample Render")

# From microfacets to participating media: A unified theory of light transport with stochastic geometry p/b The Tungsten Renderer #

## About ##

This is the code for our [2024 SIGGRAPH paper](https://cs.dartmouth.edu/~wjarosz/publications/seyb24from.html). It is based on the [Tungsten renderer](https://github.com/tunabrain/tungsten) and we defer to the Tungsten for general documentation.

A Gaussian process implicit surface is implemented as a new medium type, which allows for free-flight distance sampling. We use a track length estimator to compute transmittance.
To account for "surface-like reflectance" at what are technically medium scattering events, we implement a new phase function that defers to a BSDF instance for scattering computations.
We implement our own Gaussian process utilities.
Spatially varying mean and covariance parameters can be specified in many ways, including via dense or OpenVDB-based grids.

In addition to the core code, `src/tungsten-test` contains many experiments and test executables.
