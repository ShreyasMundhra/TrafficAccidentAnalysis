# TrafficAccidentAnalysis

## NY Collisions
### Data cleaning 
#####.1
Plot dots on NY Map, fill in the blanks in "Borough" according to the latitude and longitude of it.

#####.2
number of persons injured: max 16, min 0, average 0.277962584
number of persons killed: max 3, min 0, average 0.001007111
number of pedestrians injured: max 7, min 0, average 0.045564135
number of pedestrians killed: max 2, min 0, average 0.000488296
number of cyclist injured: max 4, min 0, average 0.027726066
number of cyclist killed: max 1, min 0, average 0.000076296
number of motorist injured: max 14, min 0, average 0.205114902
number of motorist killed: max 1, min 0, average 0.000457778

#####.3
Add a new column to count the number of vehicles contribute to a certain event.

#####.4
Accident reason prediction: some accidents' contributing factors have been
missed, and we can fill it with traning a prediction model.
Training set: number of associated vehicles, reason, date, location
Prediction set: number of associated, date, location 


