from os import path
from math import sin, cos, atan2, sqrt

from run import Predictor
from util import *


# Calculate distance between Latitude/Longitude points
# Reference: http://www.movable-type.co.uk/scripts/latlong.html
EARTH_RADIUS = 6371e3
def dist(lat1, lon1, lat2, lon2):
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	
	a = sin(dlat / 2.0) * sin(dlat / 2.0) + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0)
	c = 2 * atan2(sqrt(a), sqrt(1-a))
	
	d = EARTH_RADIUS * c
	return d


# Evaluate L1 distance on valid data
def evaluateYearbook():
	test_list = listYearbook(False, True)
	predictor = Predictor()
	
	total_count = len(test_list)
	l1_dist = 0
	print "Total testing data", total_count
	for image_name in test_list:
		image_path = path.join(YEARBOOK_PATH, image_name)
		pred_year = predictor.predict(image_path)
		truth_year = label(image_name)
		l1_dist += abs(pred_year - truth_year)
	print "L1 distance", l1_dist
	return l1_dist	
		
# Evaluate L1 distance on valid data
def evaluateStreetview():
	test_list = listStreetView(False, True)
	predictor = Predictor()
	
	total_count = len(test_list)
	l1_dist = 0
	print "Total testing data", total_count
	for image_name in test_list:
		image_path = path.join(STREETVIEW_PATH, image_name)
		pred_lat, pred_lon = predictor.predict(image_path)
		truth_lat, truth_lon = label(image_name)
		l1_dist += dist(pred_lat, pred_lon, truth_lat, truth_lon)
	print "L1 distance", l1_dist
	return l1_dist	

if __name__ == "__main__":
    evaluateYearbook()
    #evaluateStreetview()
    
