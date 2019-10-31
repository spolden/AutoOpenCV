import cv2
import os
import dlib
import shutil

picpath = input("Input your postive-samples' path: ")
if not os.path.isdir(picpath):  # 无文件夹时创建
   print('No such file or directory')
   exit(0)

badfilepath = input("Input where you want to put the invalid file: ")
if not os.path.isdir(badfilepath):  # 无文件夹时创建
    os.makedirs(badfilepath)

savepath = input("Input where you want to put the pre-approached samples: ")
if not os.path.isdir(savepath):  # 无文件夹时创建
    os.makedirs(savepath)

negpath = input('Input the path to your negative samples: ')
if not os.path.isdir(negpath):  # 无文件夹时创建
    os.makedirs(negpath)

vecname = input('Input your name of .vec file: ')+'.vec'

cascadepath = input('Input where you want to put the cascade: ')

maxfalsealarmrate = input('MaxFalseAlarmRate:')

minhitrate = input('MinHitRate: ')

stages = input('Numstages: ')

type = input('FeatureType:')

tread = input('Number of treads: ')

#if feature == '1':
#    type = 'HOG'
#if feature == '2':
#    type = "HAAR"
#if feature == '3':
#    type = "LBP"
#else:
#    print('Please enter 1 ,2 or 3')
#    exit(0)

if not os.path.isdir(cascadepath):
    os.makedirs(cascadepath)
def datawashing():
    detector = dlib.get_frontal_face_detector()
    filelst = os.listdir(picpath)
    cnt = 0
    for lst in filelst:
        read = cv2.imread(picpath + '/' + lst)
        gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
        if read is not None:
            det = detector(gray, 1)
            for i, d in enumerate(det):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = gray[x1:y1, x2:y2]
                cv2.resize(face, (64, 64))
                cv2.imwrite(savepath + '/' + str(cnt)+'.jpg', face)
                cnt = cnt + 1

        else:
            shutil.move(picpath + '/' + cnt, badfilepath)
            print(lst + '' + 'is a bad or invalid image file!!!' + 'Find it in' + ' ' + badfilepath)
    print('Useful samples: ' + str(cnt))
    return cnt


def generatevec():
    filelst = os.listdir(negpath)
    for lst in filelst:
        read = cv2.imread(negpath+'/'+lst)
        gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(negpath+'/'+lst,gray)
    #if not os.path.isdir('temp'):  # 无文件夹时创建
    #    os.makedirs('temp')
    #else:
    #    pass
    #if not os.path.isfile(vecname+'.vec'):
    #    f1 = open('temp/'+vecname+'.vec', mode="w", encoding="utf-8")
    #    f1.close()
    #else:
    #    pass
    poslst = os.listdir(savepath)
    neglst = os.listdir(negpath)
    negcnt = 0
    if os.path.isfile('pos.txt'):
        os.remove('pos.txt')
    if os.path.isfile('neg.txt'):
        os.remove('neg.txt')
    pos1 = open('pos.txt', mode="a", encoding="utf-8")
    neg1 = open('neg.txt', mode="w", encoding="utf-8")
    for pos in poslst:
        poswrt = savepath+'/'+pos+' 1 0 0 64 64'+"\n"
        pos1.write(poswrt)
    for neg in neglst:
        negwrt = negpath+'/'+neg+"\n"
        neg1.write(negwrt)
        negcnt = negcnt+1
    pos1.close()
    neg1.close()
    if os.path.isfile(vecname):
        os.remove(vecname)
    command = 'opencv_createsamples'+' -vec '+vecname+' -info '+'pos.txt'+' -bg '+'neg.txt'+' -w '+'64 '+' -h '+'64 '+'-num '+str(countpos)
    os.system(command)
    return negcnt

def training():
    command = 'opencv_traincascade'+' -data '+cascadepath+' -vec '+vecname+' -bg neg.txt'+' -numStages '+str(stages)+' -numPos '+str(countpos)+' -numNeg '+str(generatevec())+' -featureType '+type+' -w 64 -h 64 '+' -minHitRate '+str(minhitrate)+' -maxFalseAlarmRate '+str(maxfalsealarmrate)+' -numTreads '+str(tread)
    print('Training STARTS here')
    print('Please PAY ATTENTION to the LOAD')
    os.system(command)

countpos = datawashing()
generatevec()
training()