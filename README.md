# modWorm

modWorm is a generic modeling framework for modular simulation of neural connectomics, dynamics and biomechanics of Caenorhabditis elegans (C. elegans).
\
\
modWorm enables fully modular construction and simulation of C. elegans neural connectomics, dynamics, biomechanics and the environment. modWorm employs a modular hierarchical modeling pipeline where users can freely combine smaller models (e.g., individual biophysical processes) to construct larger models (e.g., nervous system) in an efficient and intuitive manner. 
\
\
The constructed model is customizable and is reusable. The underlying components can be edited or exchanged within the framework interface. modWorm has been developed as a compact all-in-one Python library with support for Julia backbone to achieve high-performance efficient simulation. The framework currently includes scenarios for simulating the nervous system, muscles and body of C. elegans as detailed in [citation below].

# Dependencies

#### Python (>3.11 recommended):
scipy, matplotlib, statsmodels, ipython, jupyter, ffmpeg, seaborn, pyjulia
#### Julia (>1.9 recommended):
DifferentialEquations, OrdinaryDiffEq, Sundials, LinearAlgebra, LogExpFunctions, Interpolations, StatsBase

# How to cite

If you are using this package please cite the following:
\
\
[Kim, J., Florman, J. T., Santos, J. A., Alkema, M. J., & Shlizerman, E. (2025). Modular integration of neural connectomics, dynamics and biomechanics for identification of behavioral sensorimotor pathways in Caenorhabditis elegans. bioRxiv, 724328.] [LINK TO THE PAPER](https://www.biorxiv.org/content/10.1101/724328v3)
