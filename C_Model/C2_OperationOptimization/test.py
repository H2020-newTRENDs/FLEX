OnDays = 200    # Parameter from DB
Year = 365
Offdays = Year - OnDays
rest = round(200/165,2)
add = 200/165
i = 1

s = []

print(rest)

while i <= Year:
    if rest > 1:
        s.append(1)
        rest = rest-1
        #print(rest)
        i = i+1
    elif rest < 1:
        s.append(0)
        rest = rest+add
        #print(rest)
        i = i+1


print(s)        #return value of function

print(sum(s))


