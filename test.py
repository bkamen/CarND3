import csv
import os


# data import and conversion
# import csv file and create list with lines
ln = []
with open('./data_sim/driving_log.csv') as logfile:
    reader = csv.reader(logfile)
    for i in reader:
        ln.append(i)

str = ln[2][0]
str = str.split('\\')[-1]

curr_dir = os.getcwd();

print(curr_dir+'data_sim/IMG/'+str)