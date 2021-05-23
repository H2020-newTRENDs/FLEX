Tout = [3.95, 3.15, 5.12, 1.23, 2.23, 2.1, 2.97, 2.44, 4.55, 4.77]

COPtemp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
COP = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
j = 0

for i in range(0, len(list(Tout))):
    #print(Tout[i])
    for j in range(0, len(list(COPtemp))):
        if COPtemp[j] < Tout[i]:
            continue
        else: print(COP[j])
        break
