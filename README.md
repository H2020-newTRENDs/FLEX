# README

This ***prosumager*** model is developed for ***NewTRENDs Project*** and ***master thesis of Thomas***.

Hey, check this out: [Prosumager](https://songminyu.github.io/Prosumager/).



### 1 Milestones

- 01-06-2021: ready to run optimization
- 01-07-2021: finalize the optimization (code and database) and summarize the results

### 2 Next steps

#### 2.1 Songmin

- generate the hot water demand profile table
- calculate the energy demand for hot water

#### 2.2 Thomas

- try some optimization examples
- dynamic COP: update it in the database
- communicate with Philipp and start trying to set up the optimization model

#### 2.3 Philipp

- work on the "B3\_Building.py"
- update building parameter tables --> ID_BuildingOption table
- write to Giacomo for a profile with heating and cooling for validation

### 5 Questions to discuss

#### 5.1 Model optimization coverage

- smart appliance: dryer, dish washer, washing machine
- space heating
- space cooling
- PV
- battery
- EV

#### 5.2 Modeling of hot water

- Do not optimize

  > - calculate the hot water demand for each person, and relate this number to lifestyle assumption, then generate the hourly demand profile
  > - generate the hourly electricity demand profile (only the electricity consumed by the electrict heater)
  > - calculate the boiler energy consumption (from 10 to 40 degree), before electric heater?

#### 5.3 Only optimize for typical days or weeks

- generate the base electricity demand profile for representative households on typical days
- we only optimize for the typical days, but they need to be selected carefully
- based on the results of these typical days or weeks, we generate the 8760-hour operation profile of the household


