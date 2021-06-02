import pdb, os, glob, argparse, cv2
from scipy.io import wavfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description = "TrainArgs");

parser.add_argument('--root_dir', type=str, default="./voxanime/", help='');
parser.add_argument('--dev',  type=str, default="./data/dev.txt", help='');
parser.add_argument('--test',  type=str, default="./data/test.txt", help='');

args = parser.parse_args();

files = glob.glob(args.root_dir+'/pycrop/*/*')
# dev = glob.glob(args.root_dir+'/pycrop/*/*')
dev, test = train_test_split(files, test_size=0.1, random_state=1004)

g = open(args.dev,'w')

for fname in tqdm(dev):

    wavname = fname.replace('pycrop', 'pywav').replace('avi', 'wav')
#     txtname = os.path.dirname(fname.replace('pycrop', 'pytxt')) + '_offset.txt'

    ## Read offset
#     f = open(txtname,'r')
#     txt = f.readlines()
#     f.close()

#     if txt[2].split(':')[0] == 'offset':
#         offset = txt[2].split()[0].split(':')[-1]
#     else:
#         print('Skipped %s - unable to read offset'%fname)
#         continue;

    ## Read video length
    cap = cv2.VideoCapture(fname)
    counted_frames = 0
    while True:
        ret, image = cap.read()
        if ret == 0:
            break
        else:
            counted_frames += 1
    total_frames = cap.get(7)
    cap.release()

    if total_frames != counted_frames:
        print('Skipped %s - frame number inconsistent'%fname)
        continue;

    ## Read audio
    sample_rate, audio  = wavfile.read(wavname)

    lendiff = len(audio)/640 - counted_frames

    if abs(lendiff) > 1:
        print('Skipped %s - audio and video lengths different'%fname)
        continue;

    g.write('%s %s %s %d\n'%(fname,wavname,0,counted_frames))

g.close()

g = open(args.test,'w')

for fname in tqdm(test):

    wavname = fname.replace('pycrop', 'pywav').replace('avi', 'wav')
#     txtname = os.path.dirname(fname.replace('pycrop', 'pytxt')) + '_offset.txt'

    ## Read offset
#     f = open(txtname,'r')
#     txt = f.readlines()
#     f.close()

#     if txt[2].split(':')[0] == 'offset':
#         offset = txt[2].split()[0].split(':')[-1]
#     else:
#         print('Skipped %s - unable to read offset'%fname)
#         continue;

    ## Read video length
    cap = cv2.VideoCapture(fname)
    counted_frames = 0
    while True:
        ret, image = cap.read()
        if ret == 0:
            break
        else:
            counted_frames += 1
    total_frames = cap.get(7)
    cap.release()

    if total_frames != counted_frames:
        print('Skipped %s - frame number inconsistent'%fname)
        continue;

    ## Read audio
    sample_rate, audio  = wavfile.read(wavname)

    lendiff = len(audio)/640 - counted_frames

    if abs(lendiff) > 1:
        print('Skipped %s - audio and video lengths different'%fname)
        continue;

    g.write('%s %s %s %d\n'%(fname,wavname,0,counted_frames))

g.close()