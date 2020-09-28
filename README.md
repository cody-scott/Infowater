# Infowater

This repo contains a collection of tools to help build and review infowater models.

Tools have been splot into logical groupings for the functions needed. See each for a more detailed description of individual tools.

## Connectivity Tools

Tools centered around the build phase of a model, specifically focused on finding and correcting connectivity errors in input data

## Model Analysis

Tools centered around the analysis and comparison of models. These tools are useful in the building, but are also useful in creating comparisons between model scenarios, models in different scenarios and scenarios in different models.

Additionally useful for isolating slow models, noisy features and general data errors in model runtime.

## Model Build Tools

Tools centered around the build phase of a model, speficially focused on the load phase and attribute assignment stages of a model build.

## Command Line

A number of tools, such as the model comparison, can be changed and modified by calling from the command line instead of from ArcGIS. 

Most cases this won't be required, but it does provide the flexibility to change and tweak it as you'd like.
One planned approach for this is putting in a form of callback routine that would give flex to change things before/after plotting etc. This is not yet done.

Use the imp library to import the pyt file.