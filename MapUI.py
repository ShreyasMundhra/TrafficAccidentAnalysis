import gmplot

gmap = gmplot.GoogleMapPlotter(40.797, -73.93778, 16)
# lat, lng = mymap.geocode("Columbia University")

latitudes = [40.797,40.865047,40.76436,40.755756,40.67816]
longitudes = [-73.93778,-73.9242,-73.88009,-73.96479,-73.897484]
# gmap.plot(latitudes, longitudes, 'cornflowerblue', edge_width=10)
gmap.scatter(latitudes, longitudes, '#3B0B39', size=40, marker=False)
# gmap.scatter(latitudes, longitudes, 'k', marker=True)
# gmap.heatmap(heat_lats, heat_lngs)

gmap.draw("mymap2.html")