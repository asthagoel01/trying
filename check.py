path="/var/lib/jenkins/workspace/T3Job1/FinalModel.py"
with open(path,'rt') as myfile:
    for myline in myfile:
        if myline.find(".add(Conv2D")!=-1:
            print("CNN")
            break

#This file is used to check whether code contains CNN or not.