import os
import json
import random
import torchaudio as ta
import torch.nn.functional as F
import torch
import time
import glob
from accelerate.logging import get_logger

logger = get_logger(__name__)

SR=44100
SLEN = SR*10

def get_audio_files(root):
    files = []
    for fn in glob.glob(os.path.join(root, "**/*"), recursive=True):
        ext = os.path.splitext(fn)[1]
        #print(fn, ext)
        if ext in {'.m4a', '.mp3', '.opus'}:
            files.append(fn)
    return files
        
class FileIter:
    def __init__(self, data, start, step):
        self.data = data
        self.start = start
        self.step = step
        
    def has_next(self):
        return self.start + self.step < self.data.shape[1]

    def next(self):
        start = self.start
        self.start += self.step
        return self.data[:, start:start+self.step]

    @classmethod
    def get_file_iter(cls, fn, slen, shift):
        audio, sr = ta.load(fn)
        nch = audio.shape[0]
        if nch != 2:
            print(f"skip channel {nch}")
            return None
        if sr != 44100:
            print('skip sr', sr, fn)
            return None
        return FileIter(audio, shift, slen)
    

class Cursor:
    def __init__(self, fileset):
        self.fileset = fileset
        self.fiter = None
        self.alive = True
    def next(self):
        if self.fiter is None or (not self.fiter.has_next()):
            #print("open NEXT")
            self.fiter = self.fileset.open_next_file()
            
        if self.fiter is not None:
            return self.fiter.next()
        else:            
            return None
        
class FileSet:
    def __init__(self, parallel, roots, slen, shift=False,
                 shuffle=False,
                 loop=False, exclude_roots=None, seed=None):
        self.slen = slen
        self.roots = roots
        self.parallel = parallel
        if seed is None:
            seed = int(time.time())
        self.rng = random.Random(seed)
                
        self.shift = shift
        self.i = -1
        self.cur_audio = None
        self.start = 0
        self.loop = loop

        self.repeat = 0

        self.exclude = {}
        if exclude_roots:
            self.exclude = {
                os.path.basename(fn)
                for exclude_root in exclude_roots
                for fn in get_audio_files(exclude_root)
            }
        self.files = sorted([fn
            for root in roots
            for fn in get_audio_files(root)
            if os.path.basename(fn) not in self.exclude])
        self.shuffle = shuffle
        if shuffle:            
            random.shuffle(self.files)            
            
        #logger.info(f"load {len(self.files)} files, exclude {self.exclude}")
        
        self.file_i = 0
        self.cursor_i = -1
        self.cursors = [Cursor(self) for i in range(parallel)]        
        
    def open_next_file(self):
        while True:
            if self.file_i >= len(self.files) and self.loop:
                self.file_i = 0
                if self.shuffle:
                    random.shuffle(self.files)
                
            if self.file_i >= len(self.files):                
                return None
            fn = self.files[self.file_i]
            logger.debug(f"open {fn}")
            if self.shift:
                shift = self.rng.randint(0, self.slen)
            else:
                shift = 0
            fiter = FileIter.get_file_iter(fn, self.slen, shift)        
            self.file_i += 1
            if fiter is not None and fiter.has_next():
                return fiter             
            
    def next(self):                
        self.cursor_i = (self.cursor_i + 1) % len(self.cursors)
        cursor = self.cursors[self.cursor_i]
        data = cursor.next()
        if data is not None:
            return data               
        return None            

class MixDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 ds_root, main, noise,
                 cat, ex_cat,
                 shift,
                 shuffle,
                 main_dropout=0.05,
                 main_nt=5,
                 noise_dropout=0.05,
                 crop_noise=0.2,
                 noise_nt=4,
                 random_weight=True,
                ):
        slen = SR*10
        if isinstance(main, str):
            main = [main]
        if isinstance(noise, str):
            noise = [noise]
        self.random_weight = random_weight
        self.crop_noise = crop_noise

        def create_fn():
            main_fs = FileSet(main_nt, 
                               [os.path.join(ds_root, m, cat) for m in main], slen,
                               shift=shift,
                               shuffle=shuffle,
                               exclude_roots=[os.path.join(ds_root, m, ex_cat) for m in main] if ex_cat is not None else None,
                                  )
            noise_fs = FileSet(noise_nt, [os.path.join(ds_root, n, cat) for n in noise],  slen,
                                   shift=shift, shuffle=shuffle,
                                   exclude_roots=[os.path.join(ds_root, n, ex_cat) for n in noise] if ex_cat is not None else None,
                                   loop=True
                                   )
            return main_fs, noise_fs
        self.create_fn = create_fn
        self.main_dropout = main_dropout
        self.noise_dropout = noise_dropout
    def __iter__(self):
        def iterfn():
            main_fs, noise_fs = self.create_fn()
            while True:
                dropmain = random.random() < self.main_dropout
                dropnoise = random.random() < self.noise_dropout
                if dropmain and dropnoise:
                    continue
    
                if dropmain:
                    noise_seg = noise_fs.next()
                    main_seg = torch.zeros_like(noise_seg)
                else:
                    main_seg = main_fs.next()        
                    if main_seg is None:
                        return
                    if dropnoise:
                        noise_seg = torch.zeros_like(main_seg)
                    else:                    
                        noise_seg = noise_fs.next()
                if random.random() < self.crop_noise:
                    try:
                        mod = random.randint(0, 3)
                        if mod <= 1:
                            mid = random.randint(0, SLEN-1)
                            if mod == 0:
                                noise_seg[:, :mid] = 0
                            else:
                                noise_seg[:, mid:] = 0
                        else:
                            minl = int(SR/2)
                            start = random.randint(0, SLEN - minl - 1)
                            end = random.randint(start + minl, SLEN - 1)
                            if mod == 2:
                                noise_seg[:, start:end] = 0
                            else:
                                noise_seg[:, :start] = 0
                                noise_seg[:, end:] = 0
                    except Exception as e:
                        print("crop noise fail", e)
                    
                        
                if self.random_weight:
                    main_weight = random.uniform(0.4, 1.0)
                    noise_weight = random.uniform(0.4, 1.0)
                    main_seg *= main_weight
                    noise_seg *= noise_weight
                mix = main_seg + noise_seg
                yield mix, main_seg, noise_seg
        return iterfn()