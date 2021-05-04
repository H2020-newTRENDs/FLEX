# README

This ***prosumager*** model is developed for ***NewTRENDs Project*** and ***master thesis of Thomas***.



### 1 Milestones

- set up the code and database by the end of May (i.e. ready to run the optimization)

### 2 Next steps

#### 2.1 Songmin

- keep working on the files in "B\_Classes" and "C1\_TableGenerator"
- Set up one example for start and let Philipp know, to set up the one for building class.
- update the database

#### 2.2 Thomas

- try some optimization examples
- literature review
- dynamic COP: pre-calculate a table like electricity price (see the excel). Songmin will set up the table structure and Thomas will fill in the numbers.

#### 2.3 Philipp

- get familiar with the code and database
- work on the "B3\_Building.py"
- maybe change the calculation with other parameters (-- please let Songmin know which parameters are used)

### 3 Coding Conventions

#### 3.1 Naming convention

- ***variable***: **V**ariable**N**ame
- ***parameter defined in a function***: **p**arameter_**n**ame
- ***function***: **calc_B**uilding**S**pace**H**eating**C**ost
- ***table***: register in the REG, then use it in the code

#### 3.2 Commiting to Github

- First, pull when you start working and make sure you are working on the latest version.
- Second, pull again before pushing, because during your working time, the others may have pushed. However, even though that is the case, the revisions should be merged into your local version automatically, as long as we didn't work on the same row. If you do see the conflicts, solve it locally. ***Always pull before pushing, so that we solve the conflicts locally.***
- Third, after solving the conflict, pull again (it should be "already up to date"), and push super-fast! Hahaha

#### 3.3 Other conventions

- If you need something from the "Infrastructure", tell Songmin to create them.

### 4 Git Commands

- git pull origin master
- git add .
- git commit -m"***comments for this commit***"
- git push origin master

### 5 Questions

#### 5.1 Hot water

- Demand profile

  > From INVERT, we have demand in kWh/m2. Then, we find **average persons living in each building type** and translate it to kWh/m2 for each person. Then, based on HOTMAP data, we allocate the annual demand to hourly profiles, and at the same time, we have distinction between working days and holidays.

- Optimization

  > - water is taken from same tank of space heating, then heated up higher with electric heater.
  > - the energy of electric heater is decided by the tank temperature, but we can assume it's constant for simplication later. Then, in the space heating optimization, hot water is an exogenous heat loss of the tank.

#### 5.2 Base year for all?

- No hurry. Come back to it when working on yearly investment simulation part.


#### 5.3 Building parameter

- when Philipp decides what calculation method to use, we will know which parameters are relevant and to be collected.













