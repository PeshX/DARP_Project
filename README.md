## Getting started
The organization of the project is the following:
- instance_gen: folder in which there is a package for the generation of the instance for a complete run of the source code
- plots: empty folder which will be filled with meaninful plots, generated and overwritten at each run of the source code
- solver: folder in which there are all the packages which deals with heuristic (GA) and operative functions
- main.py: main file which ensembles all the content of the packages within a project

## Parameters
In main.py, from line 9  to lin x the parameters for the execution of the code can be found, regarding both instance generation and GA parameters.

> [!WARNING]
> The project works with every possible combination of meaningful parameters. Nevertheless, no exception was handled in case of ill-posed parameters which will results in bad performance of the algorithm.
> A fully functioning example is already provided, considering a reasonable run-time of x minutes (can vary accoridng to PC).
> Lastly, only a coarse tuning of the parameters has been done, hence the project will always work but may return sub-optimal performance due to some numerical settings in its hidden functions (like the weights of the graph or the time requests of the passengers)

Current values are:
1. rr
2. rirt


## Execution of the code
The code is ready-to-run and right after it has been correctly downloaded 

## Results and Generated plots
The only parameter to be set up by the user is 'user_path'
