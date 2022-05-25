# README

A long journey between Philipp, Thomas, and Songmin.

### Refactoring points

- Color: https://florian-dahlitz.de/articles/why-you-should-use-more-enums-in-python
- 

### Next steps and ideas

- Integrate Reference 
- Integrate Boiler: Gas, Oil etc. 
- Integrate gasoline vehicle
- Option V2G?

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

  - Question: Should we decrease the indoor temperature band width in Summer and winter to 3°C? Right now, the building
       cam be preheated from 20 to 27°C which is not very comfortable in winter. 

- Install a shading system:
    - in the model that will reduce the solar gains whenever the temperature would rise above 
      the maximum indoor temperature in the reference model. (not tested jet)
    - reduce the radiation gains by 70% whenever the solar gains are above a certain threshold?
    Drawback: Very general decision for manual/automatic shading system. If we want to include a "smart shading system"
            later, it will not be possible. 

      
- PV generation:
  - right now the PV systems are all optimally installed to create maximum generation over a year. This way we have 
  - a very high peak generation at 1pm over a whole country. I wonder if we should also consider PV generation profiles 
  - where PV systems are oriented east/west. Not sure though how to integrate that into the model as it will affect 
  - single results for each household.


- Should we increase the output temperature of the heat pump when the DHW storage or the heating storage are charged?
      this would make the operation of the tanks more realistic and decrease their efficiency. I read in several 
      papers that the usage of storages will result in higher demand because you have to charge the storage with a 
      higher temperature level. Not sure though how easy it would be to implement this in our model.

### Resources

#### Repos on the Github
 - nomenclature: https://github.com/IAMconsortium/nomenclature
 - pyam: https://github.com/iamconsortium/pyam

#### Papers and reports
 - Household profile dataset (Nature publication): 
    - https://www.nature.com/articles/s41597-022-01156-1
    - https://zenodo.org/record/5642902
 - SPF in correlation to the age of building: https://www.agora-energiewende.de/fileadmin/Projekte/2016/Sektoruebergreifende_EW/Waermewende-2030_WEB.pdf
 