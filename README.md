# README

This ***prosumager*** model is developed for ***NewTRENDs Project*** and ***master thesis of Thomas***.

Hey, check this out: [Prosumager](https://songminyu.github.io/Prosumager/).



### 1 Milestones

- 01-06-2021: ready to run optimization
- 01-07-2021: finalize the optimization (code and database) and summarize the results

### 2 Next steps

#### 2.1 Songmin

- finished

#### 2.2 Thomas

- try some optimization examples
- dynamic COP: update it in the database

#### 2.3 Philipp

- work on the "B3\_Building.py"
- update building parameter tables --> ID_BuildingOption table
- write to Giacomo for a profile with heating and cooling for validation

### 5 Questions to discuss

#### 5.1 Literature review

- Summarize the aspects that are optimized in our model and compare it with existing studies

  > - smart appliance
  > - space heating
  > - space cooling --> like heating, also optimized
  > - hot water --> totally separated from heating and not optimized
  > - PV
  > - battery
  > - EV

- Apart from covering more aspects in the optimization, are there other contributions?

  > - data resolution?
  > - geographic coverage?
  > - typical days only? ---> Giacomo profile (this should defined in detail with Thomas)

#### 5.2 Define the functions

- have a look at the updated classes and database
- go over the optimization process together and define functions in each class

#### 5.3 Hot water

- Demand profile (exogenous, not optimized)

  > - From INVERT, we have demand in kWh/m2. Then, we find **average persons living in each building type** and translate it to kWh/m2 for each person. Then, based on HOTMAP data, we allocate the annual demand to hourly profiles, and at the same time, we have distinction between working days and holidays.
  > - **Songmin**: I briefly compared relevant parameters from three sources - (1) INVERT; (2) calculation by my colleague at ISI; (3) 60kg per day. We can have a look together in the next meeting.




