 
import json
import random
data = {}
data["intents"] = []
count = 0


obj ={}
obj["patterns"] = []
obj["responses"] = []
obj["tag"] = []
with open("qa_Appliances.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj["patterns"].append(row["question"])
            obj["responses"].append(row["answer"])
            count = count +1
            if count> 100:
                break
        except:
            continue
count = 0
obj["tag"] = "Appliences"


count  = 0
obj1 ={}
obj1["patterns"] = []
obj1["responses"] = []
obj1["tag"] = []
with open("qa_Automotive.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj1["patterns"].append(row["question"])
            obj1["responses"].append(row["answer"])
            count = count +1
            if count > 100:
                break
        except:
            continue
count  = 0
obj1["tag"] = "Automotive"


count  = 0
obj2 ={}
obj2["patterns"] = []
obj2["responses"] = []
obj2["tag"] = []
with open("qa_Beauty.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj2["patterns"].append(row["question"])
            obj2["responses"].append(row["answer"])
            count = count +1
            if count > 100:
                break
        except:
            continue
count  = 0
obj2["tag"] = "baby"


count  = 0
obj3 ={}
obj3["patterns"] = []
obj3["responses"] = []
obj3["tag"] = []
with open("qa_Beauty.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj3["patterns"].append(row["question"])
            obj3["responses"].append(row["answer"])
            count = count +1
            if count > 100:
                break
        except:
            continue
obj3["tag"] = "Beauty"


count  = 0
obj4 ={}
obj4["patterns"] = []
obj4["responses"] = []
obj4["tag"] = []
with open("qa_Clothing_Shoes_and_Jewelry.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj4["patterns"].append(row["question"])
            obj4["responses"].append(row["answer"])
            count = count +1
            if count > 100:
                break
        except:
            continue
count  = 0
obj4["tag"] = "Clothing"

count  = 0
obj5 ={}
obj5["patterns"] = []
obj5["responses"] = []
obj5["tag"] = []
with open("qa_Electronics.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj5["patterns"].append(row["question"])
            obj5["responses"].append(row["answer"])
            count = count +1
            if count > 100:
                break
        except:
            continue
obj5["tag"] = "Electronics"

count  = 0
obj6 ={}
obj6["patterns"] = []
obj6["responses"] = []
obj6["tag"] = []
with open("qa_Health_and_Personal_Care.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj6["patterns"].append(row["question"])
            obj6["responses"].append(row["answer"])
            count = count +1
            if count > 100:
                break
        except:
            continue
obj6["tag"] = "healthcare"


count  = 0
obj7 ={}
obj7["patterns"] = []
obj7["responses"] = []
obj7["tag"] = []
with open("raw.json") as f:
    for line in f:
        try:
            line = line.replace("'",'"') 
            row= json.loads(line)
            obj7["patterns"].append(row["question"])
            obj7["responses"].append(row["answer"])
            count = count +1
            if count > 100:
                break
        except:
            continue
obj7["tag"] = "others"


data ["intents"] = [obj, obj1, obj2, obj3, obj4, obj5, obj6, obj7] 

# print(create_object('bab','raw.json', 'baby'))


with open('data.json', 'w') as outfile:
    json.dump(data,outfile,  sort_keys=True, indent=4, separators=(',', ': '))








        