## Getting started
The organization of the project is the following:
- **instance_gen**: folder dealing with the generation of a complete instance of the DARPT
- **plots**: folder filled with meaningful plots, generated and overwritten at each run of the source code
- **solver**: folder in which there are all the packages which deals with heuristic (GA) and operative functions
- <ins>main.py</ins>: main file which ensembles all the content of the packages

## Parameters
In main.py, from line 9 to line 21, the parameters for the execution of the code and their meaning can be found, regarding both instance generation and GA parameters.

> [!WARNING]
> The project works with every possible combination of meaningful parameters. In case of ill-posed parameters regarding the creation of Passengers-Transfer instance (ex. 100 passengers and 1 transfer, considering capacity of a transfer ranging from 5 to 10), an exception will be raised, stopping the execution of the code.\
> A fully functioning example is already provided (same settings used for obtaining the results diplayed in the report), considering a run-time of approximately 30 minutes (can vary according to the PC).\
> Lastly, only a coarse tuning of the parameters has been done, hence the project will always work but may return sub-optimal performance due to some numerical (not parametrical) settings in its hidden functions (like the weights of the graph or the time requests of the passengers).

## Execution of the code
The code is ready-to-run and right after it has been correctly downloaded it can be used. The only parameter to be necessarily set up by the user is 'user_path', which is the local path in which the folder 'plot' is located on the user's PC. 

> [!NOTE]
> The path shall be given in this way: r'C:\Users\folder1\folder2\GitHub\DARP_Project\plots'.\
> If the 'r' is not present before the path, the code will not start.

## Results and Generated plots
Once the execution of the code has ended, the results regarding the overall fitness values of the GA will be available in the 'plot' folder. Since we implemented two selection operators (roulette_wheel and tournament), the name of the figures will be 'plotX_r' if it regards the roulette while 'plotX_t' if it regards the tournament, with X going from 1 to 4. Overall, 8 plots will be generated. In addition, on the terminal it will be printed:
- number of nodes in the current graph
- the best individual obtained with roulette wheel selection and its fitness
- the best individual obtained with tournament selection and its fitness
