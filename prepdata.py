import json
import random
data = {}
data["intents"] = []
count = 0
#creating an array files to prepare dataset
array_of_files = [
    "qa_Appliances.json", 
    "qa_Automotive.json", 
    "qa_Baby.json",
    "qa_Beauty.json",
    "qa_Clothing_Shoes_and_Jewelry.json",
    "qa_Electronics.json", 
    "qa_Health_and_Personal_Care.json",
    "raw.json"
]

for  i   in range(len(array_of_files)):
    count  = 0
    obj ={}
    obj["questions"] = []
    obj["answers"] = []
    #reading a file
    with open(array_of_files[i]) as f:
        for line in f:
            try:
                line = line.replace("'",'"') 
                row= json.loads(line)
                #question in the dataset
                obj["questions"].append(row["question"])
                #answers in the dataset
                obj["answers"].append(row["answer"])
                count += count
                if count > 100:
                    break
            except:
                #just continue if data has mistakes in , and special characters
                continue
    #attach a class label
    obj["catagory"] = str(i)
    #add the object 
    data["data"].append(obj)


#save the json file
with open('data.json', 'w') as outfile:
    #using proper indentation to display the correct json
    json.dump(data,outfile,  sort_keys=True, indent=4, separators=(',', ': '))