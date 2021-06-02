Leave = 7

Come = 17

list =[]

for i  in range(0, Leave):
    list.append(1)
for i in range(Leave,Come):
    list.append(0)
for i in range(Come,24):
    list.append((1))

CarAtHomeHours = list * 365
print(CarAtHomeHours)
print(len(CarAtHomeHours))