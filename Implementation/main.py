def MLAlgorithm(student = []):
	willAccept = False
	# run through ML algorithm, which will return whether or not student will accept
	return True

def allocateStudents(sortedStudents, maxCapacity): #array of arrays, int
	
	print("Sorted list of students by averages:")
	for x in range(len(sortedStudents)): 
		print(sortedStudents[x])
	
	cutoff = 0 #float
	acceptedStudents = 0 #int
	offerStudent = [] #array of student avaerages for those to accept
	
	breakingPt = 0 
	
	for student in allStudents:
		if MLAlgorithm(student): # they will accept, extend offer
			acceptedStudents+=1
			offerStudent.append(student[0])

		if acceptedStudents > cutoff:
			breakingPt = student[0] # this will be the average we stop accepting at
			#if student[]
			#print('hello')
			break
	
	return cutoff

allStudents = [
				#calc GPA, location, uni rank, school, gender, isAcceptee
				[93.56,		10,			1,		5,		0,		True],
				[85.07,		5,			3,		25,		1,		True],
				[99.87,		10,			1,		5,		0,		False]
				]

allStudents = sorted(allStudents, key=lambda x: x[0], reverse=True) # sort and store students by descending calculated average

allocateStudents(allStudents, 2)