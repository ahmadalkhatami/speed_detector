import cv2
import dlib
import time
import threading
import math
#memanggil file
carCascade = cv2.CascadeClassifier('myhaar.xml')#file model
video = cv2.VideoCapture('test.mp4')#file video
# video = cv2.VideoCapture(0) # ini buat realtime nya
#mengatur resolusi output. mengatur lebar dan tinggi bingkai output
WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	ppm = 8.8
	d_meters = d_pixels / ppm
	print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 18
	speed = d_meters * fps * 3.6
	return speed
	
#menginisialisasikan variabel dan konstanta yang diperlukan untuk melacak lokasi
def trackMultipleObjects():
	warnaBoundingBox = (0, 0, 255)#warna bounding box dalam RGB
	frameCounter = 0 #variabel frame counter
	IDsaatIni = 0 #variabel ID kendaraan
	fps = 0 #variabel fps
	
	carTracker = {}
	carNumbers = {}
	Lokasi1 = {}#lokasi objek pertamakali
	Lokasi2 = {}#lokasi objek saat ini
	speed = [None] * 1000
	
	# Write output to video file
	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))
	object_detector = cv2.createBackgroundSubtractorMOG2()# memisahkan backgroun dengan objek bergerak

	while True:
		start_time = time.time()#mengukur waktu eksekusi program
		# membaca video
		rc, image = video.read()#membaca video
		#modiv
		mask = object_detector.apply(image)#membuat video menjadi citra hitam putih
		cv2.imshow("Mask", mask)#menampilkan output dalam bentuk video hitamputih

		if type(image) == type(None):#fungsi untuk membaca video dari frame pertama sampai terakhir
			break
		
		image = cv2.resize(image, (WIDTH, HEIGHT))#mengubah ukuran image berdasarkan variabel tinggi dan lebar
		resultImage = image.copy()#membuat salinan
		
		frameCounter = frameCounter + 1#penghitung bingkai
		
		carIDtoDelete = []
		#memperbaharui tracker dengan dlib correlation tracker
		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)# variabel peak to-side lobe ratio
			
			if trackingQuality < 6:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print ('Removing carID ' + str(carID) + ' from list of trackers.')
			print ('Removing carID ' + str(carID) + ' previous location.')
			print ('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			Lokasi1.pop(carID, None)
			Lokasi2.pop(carID, None)
		#fungsi untuk memeriksa apakah variabel frameCounter adalah kelipatan dari parameter fc, jika iya jalankan classifier pada frame saat ini
		if not (frameCounter % 20):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#mengconvert video menjadi skala abu abu
			cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))#menentukan parameter untuk ukuran bounding box untuk objek yg akan di deteksi
			
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				#looping objek yang terdeteksi dari posisi awal dan posisi saat ini dan menghitung centroid dari abjek
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
				
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
				#membandingkan boundingbox dari objek yang terdeteksi dengan objek yang telah diprediksi oleh tracker dan menetapkan ID pada objek
				if matchCarID is None:
					print ('Creating new tracker ' + str(IDsaatIni))
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					carTracker[IDsaatIni] = tracker
					Lokasi1[IDsaatIni] = [x, y, w, h]

					IDsaatIni = IDsaatIni + 1
		
		#cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

		# mendapatkan posisi dalam bingkai saat ini
		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), warnaBoundingBox, 2)
			
			# speed estimation
			Lokasi2[carID] = [t_x, t_y, t_w, t_h]
		
		end_time = time.time()
		
		if not (end_time == start_time):
			fps = 1.0/(end_time - start_time)
		
		cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)#menampilkan FPS output

		#perhitungan kecepatan untuk objek yang terdeteksi. ketika kecepatan terdeteksi maka akan di simpan dalam variabel
		for i in Lokasi1.keys():
			if frameCounter % 1 == 0:
				[x1, y1, w1, h1] = Lokasi1[i]
				[x2, y2, w2, h2] = Lokasi2[i]
		
				# print 'lokasi sebelumnya: ' + str(Lokasi1[i]) + ', current location: ' + str(Lokasi2[i])
				Lokasi1[i] = [x2, y2, w2, h2]
		
				# print 'lokasi baru saat ini: ' + str(Lokasi1[i])
				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
						speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])


					if speed[i] != None and y1 >= 180:
						cv2.putText(resultImage, str(int(speed[i])) + " km/j", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 1)#menampilkan hasil kecepatan
						print ('ID kendaraan ' + str(i) + ': kecepatannya adalah ' + str("%.2f" % round(speed[i], 0)) + ' km/j.\n')

						print ('ID kendaraan ' + str(i) + ' Location1: ' + str(Lokasi1[i]) + ' Location2: ' + str(Lokasi2[i]) + ' kecepatannya adalah ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
		cv2.imshow('result', resultImage)



		#membuat fungsi untuk keluar looping dengan tombol esc
		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()
