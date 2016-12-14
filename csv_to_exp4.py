
f = open("fiveyearfinaldataset_Comprehensive.csv", "r")

featsstr = f.readline()

feats = featsstr[:-1].split(",")[2:]

libsvm2010 = ""
libsvm2011 = ""
libsvm2012 = ""
libsvm2013 = ""
libsvm2014 = ""
num = 0
for line in f:
    linestr = ""
    data = line.split(",")
    
    for i in range(len(feats)):
        if feats[i][:3] == "CCC" or feats[i].isupper():
            if data[2 + i] != "" and data[2 + i] != "0":
                linestr += " " + str(i +2) + ":" + data[2+i]
        if feats[i] == "secondoutlier":
            if data[2 + i] == "" or data[2 + i] == "0":
                linestr = "0" + linestr
            else:
                linestr = "1" + linestr
    if data[1] == "2010":
        libsvm2010 += linestr + "\n"
    elif data[1] == "2011":
        libsvm2011 += linestr + "\n"
    elif data[1] == "2012":
        libsvm2012 += linestr + "\n"
    elif data[1] == "2013":
        libsvm2013 += linestr + "\n"
    elif data[1] == "2014":
        libsvm2014 += linestr + "\n"

f.close()

with open("exp_4_2010.libsvm", "w") as f:
    f.write(libsvm2010)
with open("exp_4_2011.libsvm", "w") as f:
    f.write(libsvm2011)
with open("exp_4_2012.libsvm", "w") as f:
    f.write(libsvm2012)
with open("exp_4_2013.libsvm", "w") as f:
    f.write(libsvm2013)
with open("exp_4_2014.libsvm", "w") as f:
    f.write(libsvm2014)
