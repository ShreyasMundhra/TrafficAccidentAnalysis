import gmplot
import pandas as pd
import pickle
from flask import Flask, render_template
from flask.json import jsonify

app = Flask(__name__)


# gmap = gmplot.GoogleMapPlotter(40.797, -73.93778, 16)
# # lat, lng = mymap.geocode("Columbia University")
#
# latitudes = [40.797,40.865047,40.76436,40.755756,40.67816]
# longitudes = [-73.93778,-73.9242,-73.88009,-73.96479,-73.897484]
# gmap.plot(latitudes, longitudes, 'cornflowerblue', edge_width=10)
# gmap.scatter(latitudes, longitudes, '#3B0B39', size=40, marker=False)
# # gmap.scatter(latitudes, longitudes, 'k', marker=True)
# # gmap.heatmap(heat_lats, heat_lngs)
#
# gmap.draw("mymap.html")

@app.route('/')
def index():
	return render_template('mymap2.html')


@app.route('/my-link/')
def my_link():
	print 'I got clicked!'
	return 'Click.'


@app.route('/submit-click/', methods=['POST'])
def submit_click():
	print 'I got submitted!'
	return 'Submit.'


def countColValues(df, cols):
	countDict = {}
	for index, row in df.iterrows():
		# for i in range(1, 6):
		for col in cols:
			# factor = row['CONTRIBUTING FACTOR VEHICLE ' + str(i)]
			factor = row[col]
			if (not pd.isnull(factor)):
				if (factor in countDict.keys()):
					countDict[factor] = countDict[factor] + 1
				else:
					countDict[factor] = 1

	# for w in sorted(factorCount, key=factorCount.get, reverse=True):
	#     print(w, factorCount[w])
	return countDict


def getCollisionsDf():
	print("Inside get collisions df")
	return pd.read_csv("ProcessedData/NYPD_Motor_Vehicle_Collisions.csv", low_memory=False)


# @app.route('/build-dict/', methods=['GET'])
def buildAccidentWithLocationsDict():
	print("Inside initialize")
	df = getCollisionsDf()
	df = df[0:20000]

	accidentWithLocsDict = {}
	for index, row in df.iterrows():
		print(index)
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

@app.route('/get_locs/<accidentDict>', methods=['GET','POST'])
def getAccidentLocs(accidentDict):
	print("inside get locs")
	accidentWithLocsDict = load_obj("accident_dict")
	return jsonify({'accidentLocs': accidentWithLocsDict[accidentDict['accident']]})

@app.route('/initialize/', methods=['GET'])
def getAccidents():
	print("Inside get accidents")
	accidentWithLocsDict = load_obj("accident_dict")
	print(type(accidentWithLocsDict))
	print(accidentWithLocsDict.keys())
	return jsonify({'accidents': accidentWithLocsDict.keys()})

if __name__ == '__main__':
	app.run(debug=True)
	# accidentWithLocsDict = buildAccidentWithLocationsDict()
	# saveObject(accidentWithLocsDict, "accident_dict")
