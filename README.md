# README

A long journey between Philipp, Thomas, and Songmin.

### Refactoring points

- Color: https://florian-dahlitz.de/articles/why-you-should-use-more-enums-in-python
- 

### Next steps and ideas

- behavior generator
- investment decision modeling
- more building types, e.g. multi-family building
- building stock database
- national level aggregation
- energy community

- maximum indoor temperature limitation:
  - set a maximum temperature when the reference heating demand is > 0! (this way it is ensured that the SEMS can 
     not preheat the building in winter). When the heating demand == 0 the maximum indoor set temperature rises to 60°C. 
    Drawback: In rare cases the heating energy in Spring and Autumn drops to 0 
     when solar radiation is high, and the SEMS preheats the building up to 30°C. --> FIXED by checking if the heating 
     power drops also to zero in the following 8 hours and 3 hours before the event. If it is not the case, the set 
     temperature will not be raised to 60°C. 

  - install a shading system in the model that will reduce the solar gains whenever the temperature would rise above 
    the maximum indoor temperature in the reference model. (not tested jet)
    Drawback: Very general decision for manual/automatic shading system. If we want to include a "smart shading system"
              later, it will not be possible. 

  - install a "cooling" variable that is only used when cooling is turned off. This cooling variable does not cost
    electricity, thus when there is no AC the indoor room temperature will stay below the maximum temperature and 
    the necessary cooling energy to do so is for free. 
    Drawback: In the results the indoor temperature of those scenarios without cooling will not be realistic

DO YOU GUYS HAVE ANY IDEA HOW TO SOLVE THIS? The optimization should not be able to pre-heat the building above 
maximum indoor set temperature, at the same time, when no cooling system is adapted, the maximum indoor set temperature
has to be raised, otherwise the model becomes infeasible.


- PV generation:
  - right now the PV systems are all optimally installed to create maximum generation over a year. This way we have 
  - a very high peak generation at 1pm over a whole country. I wonder if we should also consider PV generation profiles 
  - where PV systems are oriented east/west. Not sure though how to integrate that into the model as it will affect 
  - single results for each household.

### Resources

#### Repos on the Github
 - nomenclature: https://github.com/IAMconsortium/nomenclature
 - pyam: https://github.com/iamconsortium/pyam

#### Papers and reports
 - Household profile dataset (Nature publication): 
    - https://www.nature.com/articles/s41597-022-01156-1
    - https://zenodo.org/record/5642902
 - SPF in correlation to the age of building: https://www.agora-energiewende.de/fileadmin/Projekte/2016/Sektoruebergreifende_EW/Waermewende-2030_WEB.pdf
 