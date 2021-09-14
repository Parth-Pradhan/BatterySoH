f = open("nopol.txt", "r")
Battery_name = f.readline()[0:]
index = f.readline()[0:]
f.close()

print (Battery_name[0:5])
print (index)