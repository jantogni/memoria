import pandas
import matplotlib.pyplot as plt
import pylab
import pywt

time_c	= 'time'
asset_1 = 'HPQ' 
family	= 'db2'

def input_data(str_input):
	df = pandas.read_csv(str_input)
	time =		[]
	roa_hp =	[]
	hpq_t = df[asset_1]

	for i in df[time_c]:
		time.append(i.split('-')[3])
		
	for i in range(1,len(hpq_t)):
		roa_hp.append((hpq_t[i] /hpq_t[i-1]) - 1)

	return time, hpq_t, roa_hp

def plot_data(x, y_1, y_2):
	#x -> time index
	#y_1 -> asset(t), color blue
	#y_2 -> returnOfAsset(t), color red
	fig = plt.figure()
	colors = ['b', 'c', 'y', 'm', 'r']	

	#Ploting price vs time
	fig1 = fig.add_subplot(111)
	fig1.set_title('Price vs Time')
	fig1.set_xlabel('time')
	fig1.set_ylabel('price')
	points_1 = plt.scatter(x, y_1, marker='o', color=colors[0])

	plt.show()
	
	#Ploting returns vs time
	fig2 = fig.add_subplot(111)
	fig2.set_title('returns vs Time')
	fig2.set_xlabel('time')
	fig2.set_ylabel('returns')
	points_2 = plt.scatter(x[1:], y_2, marker='o', color=colors[4])

	plt.show()

	return 0

def dwt(asset, time):
	cA, cD = pywt.dwt(asset, family)

	print len(cA)
	print len(cD)
	print len(time)
	
	#points_2 = plt.scatter(time, cA)

	plt.show()

	return 0

if __name__ == '__main__':
	time_v, asset_t, roa_t = input_data("input/2010-02-05_60seconds.csv")
	index_t =	[]

	for i in range(1,len(time_v)+1):
		index_t.append(i)

	dwt(asset_t, index_t)
	#plot_data(index_t, asset_t, roa_t)
