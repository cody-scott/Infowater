# Model Build Tools

These are tools that are expected to be used in the build phase of the model, when generating the required data flags for features.

## Closed Pipes

Produces a list of pipes that should have their initial status be set to "Closed", representing a closed valve.

Useful when working with a pressure zone feature class or a valve layer that includes a closed status on the pipe. 

## Demand Isolation Nodes

Based on your input nodes and other sources, generates a list of nodes to flag as Demand or Isolation.

Demand nodes are the target nodes to apply the allocated demand in a model, isolation is used to roughly represent isolation valves in the system, when dealing with an all pipes model.

## Junction Elevations

Re-implementation of the node elevation tool. Default Infowater tool is very slow, this is not.

## Split Watermains

Split incorrectly segemented watermains at a particular input node. This is essentially a re-creation of the ESRI tool, but it works at the lowest license level (arcview/basic), which is generally more readily available for users