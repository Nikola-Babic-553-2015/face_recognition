#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

import json
import time
import logging
# cv2 i helper:
import cv2
from helper.common import *
from helper.video import *
import sys
sys.path.append("../..")
from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.validation import KFoldCrossValidation
from facerec.serialization import save_model, load_model
from facedet.detector import CascadedDetector
#import webbrowser
import requests  

class ExtendedPredictableModel(PredictableModel):
    

    def __init__(self, feature, classifier, image_size, subject_names):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.image_size = image_size
        self.subject_names = subject_names

def get_model(image_size, subject_names):
    """ Vraca predvidajuci model
    """
    # definise Fisherface metodu:
    feature = Fisherfaces()
    #definise 1-NN klasifikator sa Euklidskim rastojanjem Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    # vraca modela :
    return ExtendedPredictableModel(feature=feature, classifier=classifier, image_size=image_size, subject_names=subject_names)

def read_subject_names(path):
    """Cita foldere iz direktorijuma

    Args:
        path: put do direktorijuma sa osobama - licima za prepoznavanje.

    Returns:
        folder_names: ime foldera.
    """
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
    return folder_names

def read_images(path, image_size=None):
    """Cita slike iz direktorijuma, menja njihovu velicinu

    Args:
        path: put do direktorijuma sa osobama - licima za prepoznavanje.
        image_size: n-torka koja se koristi pri promeni velicine slika

    Returns:
        Listu [X, y, folder_names]

            X: slike, numpy niz
            y: labele  (jedinstveni broj lica, osobe) -- lista.
            folder_names: ime foldera.
    """
    c = 0
    X = []
    y = []
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            folder_names.append(subdirname)
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                #try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (image_size is not None):
                        im = cv2.resize(im, image_size)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                #except IOError, (errno, strerror):
                #    print "I/O error({0}): {1}".format(errno, strerror)
                # except:
                #     print "Unexpected error:", sys.exc_info()[0]
                #     raise
            c = c+1
    return [X,y,folder_names]

data_dict = {'Aleksandar':'586-2015' ,'Kondic':'575-2015','Jovan':556-2015,'Andjelkovic':'587-2015','Sofija':'574-2015','Filip':'572-2015',
'Nikola':'553-2015','Vukasin':'552-2015','Vulovic':'570-2015','Andjela':'558-2015','Jelena':'566-2015','Nevena':'557-2015',
'Kosta':'561-2015','Vlada':'584-2015','Nenad':'564-2015','Kovacevic':'567-2015'} 
def open_url(name):
	#webbrowser.open("http://rtsi.pe.hu?indeks="+name)
        print data_dict[name]
	r = requests.post("http://rtsi.pe.hu/cekalica.php", data={'number': data_dict[name]})
	
class App(object):
    def __init__(self, model, camera_id, cascade_filename):
        self.model = model
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.1)
        self.cam = create_capture(camera_id)
	self.cap = cv2.VideoCapture('http://192.168.0.136:8080/test') 

    def run(self):
	open = 0
	sendind_data = None
	time1 = time.time()
	detection_dict = {}
	count_frame = 0
	#self.model.subject_names[10] = 'Nije pronadjeno lice'
        while True:
            #ret, frame = self.cam.read()
	    ret, frame = self.cap.read()
	    print ret,frame
            # smanjujemo velicinu frejma na pola:
            img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
            imgout = img.copy()
	    detect =  self.detector.detect(img)
	    count_frame += 1
	          
	
            for i,r in enumerate(detect):
                x0,y0,x1,y1 = r
		#print x0,y0,x1,y1
                # (1) Uzimamo lice, (2) Konverujemo u grayscale & (3) skaliramo velicinu na image_size:
                face = img[y0:y1, x0:x1]
                face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, self.model.image_size, interpolation = cv2.INTER_CUBIC)
                # uzimamo model:
                prediction = self.model.predict(face)[0]
		#print self.model.predict(face)
                # iscrtavamo pravougaonik:
                cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
                # ispisemo ime (ime foldera...):
                draw_str(imgout, (x0-20,y0-20), self.model.subject_names[prediction])
		if self.model.subject_names[prediction] in detection_dict:
			detection_dict[self.model.subject_names[prediction]] += 1
		else:
			detection_dict[self.model.subject_names[prediction]] = 1
	    
            cv2.imshow('videofacerec', imgout)
            # prikazi sliku & izadji na escape ili nakon 100 frejmova:
            ch = cv2.waitKey(10) & 0xFF
            if ch == 27 or count_frame == 100:

		break
	 #izracunaj najfrekventije prepoznato lice
	max_freq = max(detection_dict.values())
	max_name = None
	for name in detection_dict:
		if detection_dict[name] == max_freq:
			max_name = name
	print detection_dict		
	sum_freq = sum(detection_dict.values())
	print max_freq,sum_freq
	
	'''if max_freq >= 0.6 * sum_freq:
		sending_data = max_name'''
	sending_data = max_name
	'''else:
		
		sending_data = 'Nepoznato lice'
'''	
	print sending_data
	if open == 0:
		#open_url(data_dict[sending_data])
		open_url(sending_data)
		open = 1	
	      

if __name__ == '__main__':
    from optparse import OptionParser
    # model.pkl se koristi kao trening model predvidjanja.
    usage = "usage: %prog [options] model_filename"
    # Opcije za treniranje, promenu velcine, validaciju i postavljanje ID kamere:
    parser = OptionParser(usage=usage)
    parser.add_option("-r", "--resize", action="store", type="string", dest="size", default="100x100",
        help="Menja velicinu datog skupa podataka u velicinu oblika[width]x[height] (default: 100x100).")
    parser.add_option("-v", "--validate", action="store", dest="numfolds", type="int", default=None,
        help="Izvrsava validaciju skupa podataka(default: None).")
    parser.add_option("-t", "--train", action="store", dest="dataset", type="string", default=None,
        help="Trenira skup podataka.")
    parser.add_option("-i", "--id", action="store", dest="camera_id", type="int", default=0,
        help="Postavlja ID kamere (default: 0).")
    parser.add_option("-c", "--cascade", action="store", dest="cascade_filename", default="haarcascade_frontalface_alt2.xml",
        help="postavlja put di Haar Cascade namenjen detekciji lica (default: haarcascade_frontalface_alt2.xml).")
    # prikazuje opcije korisnike:
    parser.print_help()
    print ("Pritisni [ESC] za izlaz!")
    print ("Script output:")
    # Parsiraj argumente:
    (options, args) = parser.parse_args()
    # proveri da li je ime modela prosledjeno:
    if len(args) == 0:
        print ("[Error] Model predvidanja nije dat.")
        sys.exit()
    # Ovaj model ce biti koriscen (ili kreiran ako  (-t, --train) postoji:
    model_filename = args[0]
    # Proveri da li postoji model, ako nije prosledjen skup podataka:
    if (options.dataset is None) and (not os.path.exists(model_filename)):
        print ("[Error] Model nije nadjen '%s'." % model_filename)
        sys.exit()
    # proveri da li postoji cascade fajl:
    if not os.path.exists(options.cascade_filename):
        print ("[Error] Cascade fajl nije nadjen '%s'." % options.cascade_filename)
        sys.exit()
    # Menjamo velcinu slika, jer algoritam to zahteva:
    try:
        image_size = (int(options.size.split("x")[0]), int(options.size.split("x")[1]))
    except:
        print ("[Error] Nije moguce parsirati podatke'%s'. Prosledite podatke u formatu [width]x[height]!" % options.size)
        sys.exit()
    # skup podataka za ucenje:
    if options.dataset:
        # proveri da li podaci postoje:
        if not os.path.exists(options.dataset):
            print ("[Error] Nije nadjen skup podataka'%s'." % dataset_path)
            sys.exit()
        # Cita slike, labele i  imena direktorijuma iz datog skupa. Slike su promenjene velicine:
        print ("Ucitavanje podataka...")
        [images, labels, subject_names] = read_images(options.dataset, image_size)
        # Recnik {labela,ime}:
        list_of_labels = list(xrange(max(labels)+1))
        subject_dictionary = dict(zip(list_of_labels, subject_names))
        # Model za procesiju:
        model = get_model(image_size=image_size, subject_names=subject_dictionary)
       #omogucava validaciju pre detekcije i prepoznavanja:
        if options.numfolds:
            print ("Validacija modela sa %s ..." % options.numfolds)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger = logging.getLogger("facerec")
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            # Validacija:
            crossval = KFoldCrossValidation(model, k=options.numfolds)
            crossval.validate(images, labels)
            crossval.print_results()
	#print crossval
        # Racunaj model:
        print ("Procesija modela...")
        model.compute(images, labels)
        #cuvamo model:
        print ("Cuvanje modela...")
        save_model(model_filename, model)
    else:
        print ("Ucitavanje modela...")
        model = load_model(model_filename)
   #prekini rad, ako nije prosledjen model
  
    if not isinstance(model, ExtendedPredictableModel):
        print ("[Error] Model nije tipa '%s'." % "ExtendedPredictableModel")
        sys.exit()
    print ("Pokretanje aplikacije...")
    App(model=model,
        camera_id= 0,
        cascade_filename=options.cascade_filename).run()
#options.camera_id
