N = 12
c = 1
interval = c/(N-1)
c = c + interval
for i in range(N//2):
    c=c-interval
    print(c)

for i in range(N//2):
    c=c-interval
    print(c)



# for i in range(N//2):
#     c2=((c/2)-(i+1/N))
#     print(c2)