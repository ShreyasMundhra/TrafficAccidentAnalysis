import pandas as pd
import pickle
from flask import Flask, render_template
from flask.json import jsonify

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('map.html')

def countColValues(df, cols):
	countDict = {}
	for index, row in df.iterrows():
		for col in cols:
			factor = row[col]
			if (not pd.isnull(factor)):
				if (factor in countDict.keys()):
					countDict[factor] = countDict[factor] + 1
				else:
					countDict[factor] = 1
	return countDict


def getCollisionsDf():
	return pd.read_csv("ProcessedData/NYPD_Motor_Vehicle_Collisions.csv", low_memory=False)

def buildAccidentWithLocationsDict():
	print("Inside initialize")
	df = getCollisionsDf()
	accidentWithLocsDict = {}
	for index, row in df.iterrows():
		latitude = row['LATITUDE']
		longitude = row['LONGITUDE']
		for i in range(1, 6):
			factor = row['CONTRIBUTING FACTOR VEHICLE ' + str(i)]
			if (not pd.isnull(factor)):
				if (factor in accidentWithLocsDict.keys()):
					accidentWithLocsDict[factor] = accidentWithLocsDict[factor].union({(latitude, longitude)})
				else:
					accidentWithLocsDict[factor] = {(latitude, longitude)}

	return accidentWithLocsDict

def saveObject(obj, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

@app.route('/get_locs/<accident>', methods=['GET','POST'])
def getAccidentLocs(accident):
	accidentWithLocsDict = load_obj("accident_dict")
	return jsonify(locs=list(accidentWithLocsDict[accident]))

@app.route('/initialize/', methods=['GET'])
def getAccidents():
	accidentWithLocsDict = load_obj("accident_dict")
	return jsonify({'accidents': accidentWithLocsDict.keys()})

if __name__ == '__main__':
	# RUN THIS FIRST
	# accidentWithLocsDict = buildAccidentWithLocationsDict()
	# saveObject(accidentWithLocsDict, "accident_dict")

	app.run(debug=True)
