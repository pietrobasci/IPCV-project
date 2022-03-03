from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:

	def __init__(self, maxDisappeared=20, entranceLine=200, maxObjectShift=300):

		self.count = 0
		self.maskcount = 0
		self.nomaskcount = 0
		self.undefined = 0

		self.nextObjectID = 0
		# dictionary that records objects detected as key-value pair (objectID-centroidCoordinates)
		self.objects = OrderedDict()
		# dictionary that records if objects are inside as a key-value pair (objectID-value)
		# value can be 0 (if it is outside), 1 (if gets inside) or -1 (if already inside)
		self.inside = OrderedDict()
		# dictionary that records if objects is with mask as a key-value pair (objectID-value)
		# value is increased by 1 if a mask is detected, decreased by 1 otherwise
		self.withmask = OrderedDict()
		# dictionary that records if objects is disappeared as a key-value pair (objectID-value)
		# value is the number of consecutive frame in which the object is not present
		self.disappeared = OrderedDict()

		# number of maximum consecutive frames a given object is allowed to be marked as "disappeared"
		# until we need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared
		self.entranceLine = entranceLine
		self.maxObjectShift = maxObjectShift

	def register(self, centroid, mask):
		# register an object using the next available objectID
		# to store the centroid
		self.objects[self.nextObjectID] = centroid
		# if the object is outside set the variable to 0, else set to -1
		# to exclude it from the count
		if centroid[1] < self.entranceLine:
			self.inside[self.nextObjectID] = 0
		else:
			self.inside[self.nextObjectID] = -1
		self.withmask[self.nextObjectID] = mask
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# update counters before deregister the object if it crossed the line
		if self.inside[objectID] == 1 and self.withmask[objectID] > 0:
			self.maskcount += 1
		elif self.inside[objectID] == 1 and self.withmask[objectID] < 0:
			self.nomaskcount += 1
		elif self.inside[objectID] == 1 and self.withmask[objectID] == 0:
			self.undefined += 1
		# deregister an object ID by deleting the object ID from
		# all the respective dictionaries
		del self.objects[objectID]
		del self.inside[objectID]
		del self.withmask[objectID]
		del self.disappeared[objectID]

	def update(self, rects, masks):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		# loop over the bounding box rectangles
		for (i, (x, y, w, h)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int(w / 2) + x
			cY = int(h / 3) + y
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], masks[i])

		# otherwise, we are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			# compute the euclidean distance between each pair of object
			# centroids and input centroids, respectively
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value is at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue
				# otherwise, grab the object ID for the current row,
				# set its new centroid, set the variable inside to 1 and
				# increase the count if it crossed the line,
				# and reset the disappeared counter
				if D[row][col] < self.maxObjectShift:
					objectID = objectIDs[row]
					self.objects[objectID] = inputCentroids[col]

					if self.objects[objectID][1] > self.entranceLine and self.inside[objectID] == 0:
						self.inside[objectID] = 1
						self.count += 1

					self.withmask[objectID] += masks[col]
					self.disappeared[objectID] = 0
					# indicate that we have examined each of the row and
					# column indexes, respectively
					usedRows.add(row)
					usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# update the disappeared counter for the unused centroid
			# and deregister if it is greater than that the maxDisappeared
			for row in unusedRows:
				# grab the object ID for the corresponding row
				# index and increment the disappeared counter
				objectID = objectIDs[row]
				self.disappeared[objectID] += 1
				# check to see if the number of consecutive
				# frames the object has been marked "disappeared"
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# register all the new input centroid
			for col in unusedCols:
				self.register(inputCentroids[col], masks[col])

		# return the set of trackable objects
		return self.objects
