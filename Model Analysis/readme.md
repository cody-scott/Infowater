# Model Analysis

These tools should be helpful in the running and validation of a model.

Expected use is centered around running one or many model scenarios and creating a logical comparison for use in calibration or reporting.

## Analyze Model Report

Produce an excel file that contains a model report analysis. This should break out the information containing in the reporting .rpt file of an infowater output.

Broken into the components of Changes, Trials, Warnings, Balance representing the components generated at each time step

## Convert Model Report

Convert the output from the analyze model report tool to an arcgis table

## Create Comparison Template

Generate the template files to be used in the Model Comparison and Model DTW Tools.

The tools need the input data to follow a specific structure, so this tool generates the excel document in the structure they must follow.

Creates both the template comparison excel document as well as the comparison scada document.

## Model Comparison

Generates a comparison between input scenarios based on your supplied comparison document. 

Key comparison point is that the data is loaded from the model output, which is set by enabling the "Export EXT_APP_FILTER" setting in infowater, as well as defining the appropriate selection set.

Based on this selection set and the comparison excel file, the tool will generate a series of graphs for the comparison of the documents.

Finally, you can choose to input SCADA data to display for comparison purposes, as well as the averaging function to apply. 

Read the help/code for further options.

## Model DTW

Implementation of Dynamic Time Warping to determine the fit between two modelled scenarios. Useful for quickly determining the difference between scenarios and input sources.

Produces DTW results comparing all input sources in the comparison excel document

Read about DTW [Wikipedia](https://en.wikipedia.org/wiki/Dynamic_time_warping)

DTW is taken from [FastDTW](https://github.com/slaypni/fastdtw/blob/master/fastdtw/fastdtw.py), primarily to avoid needing to install additional packages.


