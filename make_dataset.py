


#RCWO

#this is what makes the actual training set - stores images into 
#whatever bin size size sequences


import zipfile
import pygrib as pg
import cv2 as cv
import pickle
import numpy as np
import os

name0 = 'MSG4-SEVI-MSGCLMK-0100-0100-20220414043000.000000000Z-NA.zip'
prefix = 'MSG4-SEVI-MSGCLMK-0100-0100-'
suffix = '.000000000Z-NA'


mon = '04'
day = 5
counter = 0
bsz = 16 #~>4 hour blocks
ims = []
outpath = './data'

if os.path.isdir(outpath)==False:
    os.mkdir(outpath)


while ((mon!='05')&(day>4))==1: #stop exactly may 4th
    
    if day > 30: #end of april
        mon = '05'
        day = 1
    
    if day < 10:
        daystr = '0'+str(day)
    else:
        daystr = str(day)
    
    print(mon,daystr)
    
    for hour in range(24):
        if hour <10:
            hstr = '0'+str(hour)
        else:
            hstr = str(hour)
                
        for time in ['00','15','30','45']:
            
            counter+=1    
            name = prefix + '2022'+mon+daystr+hstr+time+'00'+suffix
            zpath = './'+name+'.zip'
            dpath = './'+name
            
            
            if counter > bsz:
                f = open(name+'.pckl',"wb") #have to call it something - this is easy                            
                pickle.dump(np.array(ims),f)
                
                ims = []
                counter = 1
                print('saved')
            
            with zipfile.ZipFile(zpath, 'r') as zip_ref:
                zip_ref.extractall(dpath)
                
            fpath = dpath+'/'+name+'.grb'
        
            grbs = pg.open(fpath)
    
            g = grbs.select(name='Cloud mask')[0]
            im,_,_ = g.data()
            im = cv.resize(im,(500,500))
            ims.append(im)
    
    day+=1
    
    #^^^ le image


#2022 05 02 23 4500
#y #mon #day #h #time




         #vv date
#2022 04 14 04 30 00
               #^^ 00,15,30,45
