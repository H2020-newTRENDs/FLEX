# Introduction to FLEX
> A Modeling Suite for the Operation and Energy Consumption of Households and Energy Communities

## Overview

FLEX is a modeling suite for the operation and energy consumption of households and energy communities.
Currently, FLEX is under development by 
**[Fraunhofer ISI](https://www.isi.fraunhofer.de/)** and 
**[TU Wien (Energy Economics Group)](https://eeg.tuwien.ac.at/)** 
in the framework of the H2020 project [newTRENDs](https://newTRENDs2020.eu/). 

FLEX contains three modules:

* `FLEX-Behavior` generates the hourly activity and energy demand profile of a pre-defined individual household. 
The results include (1) appliance electricity demand, (2) domestic hot water demand, (3) driving profile, 
and (4) building occupation.
* `FLEX-Operation` takes the results of `FLEX-Behavior` as input and calculates the household's system operation and 
energy demand in hourly resolution, given its building and technology configurations. 
The household can be calculated as (1) a consumer or prosumer (consumer + PV) without optimization, 
or (3) as a prosumagers (prosumer + energy storage + smart energy management system) with optimization.
* `FLEX-Community` takes the results of `FLEX-Operation` as input and calculates the system operation of a pre-defined 
energy community from the perspective of an aggregator, who makes profit by using following options:
(1) Facilitate the P2P electricity trading among the households in real time; and
(2) Buy the surplus electricity from the community at a lower price, save it in the battery, 
and sell it later at a higher price.

Currently, `FLEX-Operation` and `FLEX-Community` are developed and released in this repository.
`FLEX-Behavior` is under development. So, the activity and energy demand profile of households are temporally
generated based on the results from the [HOTMAPS project](https://www.hotmaps.eu).

## Key features

### FLEX-Operation
- The household's system operation and energy demand is modeled in hourly resolution. 
- The heat dynamics of the building is modeled within the 5R1C framework (DIN ISO 13790),
and the building mass is considered as thermal storage.
- Detailed and flexible configuration of a household is supported, including building, heating system 
(heat pump, fuel-based boiler, electric heater), thermal storages for space heating and domestic hot water, space 
cooling technology, PV, battery, and electric vehicle.

### FLEX-Community
- The operation of the energy community is modeled in hourly resolution.
- Based on the detailedly configured and calculated households from `FLEX-Behavior` and `FLEX-Operation`, 
users can trace back to each household and evaluate the benefit and cost of each community member.

## Getting started <div id="Getting_started"/>

To work with FLEX, please first follow the steps for installation 

1. Clone the repo to your local computer;
2. Open the project and install the requirements with `pip install -r requirements.txt` in the terminal;
3. Install a solver for the FLEX-Operation model (setup with [Pyomo](http://www.pyomo.org/)). 
We suggest to use [gurobi](https://www.gurobi.com/), and if you would like to try other solvers, 
we appreciate if you could inform us the experience. 
In the `model_opt.py` file, you can switch to others solvers in `pyo.SolverFactory("gurobi")`.

### FLEX-Operation <div id="FLEX_Operation"/>

Before running the `FLEX-Operation` model, you can check the input data in the folder `data/input_operation`. 
Then, by running the `operation_init.py` file, these files are used to initialize a project database in the folder `data/output`.
The name of the database is written in the `config.py` file, which is the `project_name` attribute. 
Opening the database file (`data/output/FLEX.sqlite`), you will find an overarching scenario table `OperationScenario`
which is set up based on the component scenario tables. Then, in the `operation_run.py` file, 
you can set which scenarios you want to run and start the model. 

The results will also be stored in the database file, including four tables: 
* OperationResult_OptimizationHour 
* OperationResult_OptimizationYear 
* OperationResult_ReferenceHour 
* OperationResult_ReferenceYear

As shown, the results are always divided into "reference" and "optimization" representing the same scenarios for
(1) consumers/prosumers (without optimization) and (2) prosumagers (with optimization). 
The tables with hourly results contain vectors of length 8760 for each variable in each scenario, 
e.g., the state of charge of the battery in each hour of the year. 
Then, the tables with yearly results contain the sum of each variable over the year, 
e.g., the total energy cost in the year. Finally, by running the `operation_ana.py` file, 
more analysis results will be generated and saved in the database and the output folder.

### FLEX-Community <div id="FLEX_Community"/>

With the results of `FLEX-Operation` saved in the database, 
you can start the `FLEX-Community` model to calculate the operation of an energy community which contains 
a set of the households that are calculated in the `FLEX-Operation` model.

First, in the `data/input_community` folder, you will find the overarching scenario table for the model. 
Then, in the `community_run.py` file, you can select 
(1) which scenarios to run by setting the `COMMUNITY_SCENARIO_IDS` variable, 
and (2) which households you want to cover in the community by setting the `OPERATION_SCENARIO_IDS` variable.
Finally, by running the `community_run.py` file, you start the model and will receive two result tables in the database file:
* CommunityResult_AggregatorHour
* CommunityResult_AggregatorYear

## Citation

If you want to use FLEX for your research, we would appreciate it if you would cite the following report, 
which contains the latest and most detailed introduction of the `FLEX-Operation` and `FLEX-Community` models.

* Yu, Songmin; Mascherbauer, Philipp; Kranzl, Lukas (2022): 
Modeling of prosumagers and energy communities in energy demand models. 
(newTRENDs - Deliverable No. D5.2). Fraunhofer ISI, Karlsruhe.

Besides, you are also welcomed to read and cite other studies from us:

* Mascherbauer, Philipp; Kranzl, Lukas; Yu, Songmin; Haupt, Thomas (2022):
Investigating the impact of smart energy management system on the residential
electricity consumption in Austria. In Energy 249, p. 123665. DOI:
10.1016/j.energy.2022.123665.
* Haupt, Thomas (2021): Prosuming, demand response and technological
flexibility: An integrated optimization model for households' energy
consumption behavior. Thesis for Master of Science. Hochschuler Ulm.


## Lisence

As mentioned, FLEX is under development by **Fraunhofer ISI** and **TU Wien (Energy Economics Group)** 
in the framework of the H2020 project [newTRENDs](https://newTRENDs2020.eu/). 
The developers (2021-2022) include:
* [Songmin Yu](https://www.isi.fraunhofer.de/en/competence-center/energiepolitik-energiemaerkte/mitarbeiter/yu.html): songmin.yu@isi.fraunhofer.de
* [Philipp Mascherbauer](https://eeg.tuwien.ac.at/staff/people/philipp-mascherbauer): philipp.mascherbauer@tuwien.ac.at
* [Thomas Haupt](https://www.hs-ansbach.de/personen/haupt-thomas/): thomas.haupt@hs-ansbach.de

FLEX is licensed under the open source [MIT License]().

