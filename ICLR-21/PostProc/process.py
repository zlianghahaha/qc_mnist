

fh = open("C8_Data","r")
lines = fh.readlines()
i=0

Dic = {}
for l in lines:
    l_list = l.strip().split("    ")
    key = l_list[0]
    val = float(l_list[1])
    if key not in Dic.keys():
        Dic[key] = []
    Dic[key].append(val)


for k,v in Dic.items():
    top_5 = v[5:]
    ave = max(top_5)
    print(k,ave)

