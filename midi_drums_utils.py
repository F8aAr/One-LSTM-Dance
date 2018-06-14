import sys
import os
import pdb
import numpy as np
import midi
from tkinter.filedialog import askdirectory
from mido import MidiFile, MidiTrack, Message, MetaMessage
from note_proc import *
import itertools

def text_to_notes(encoded_drums, note_list=None):
    '''
    0b0000 0b1000 ...  -> corresponding note.
    '''
    if note_list == None:
        note_list = Note_List()

    for word_idx, word in enumerate(encoded_drums):
        c_tick_here = word_idx*min_ppq
        if word == 'BAR':
            return note_list
        for pitch_idx, pitch in enumerate(allowed_pitch):

            if word[pitch_idx+2] == '1':
                new_note = Note(pitch, c_tick_here)
                note_list.add_note(new_note)
    return note_list

def addBar(filename,retList = None):


    f = open(filename, 'r')
    sentence = f.readline()
    drum_patterns = sentence.split(' ')
    drum_patterns_temp = []
    bar_idx = 0
    with open(filename, "w") as f:
        for word_idx, word in enumerate(drum_patterns):
            bar_idx += 1
            f.write(word + ' ')
            drum_patterns_temp.append(word)
            if bar_idx == 16:
                f.write('BAR'+' ')
                drum_patterns_temp.insert(word_idx,'BAR')
                bar_idx = 0
    f.close()
    if retList:
        return drum_patterns_temp
    else:
        return

def durBars(filename):
    f = open(filename, 'r')
    sentence = f.readline()
    drum_patterns = sentence.split(' ')
    n_loops = drum_patterns.count('BAR')
    return(n_loops)

def removeBar(filename,returnFilename = False):


    f = open(filename, 'r')
    sentence = f.readline()
    drum_patterns = sentence.split(' ')
    n_loops = drum_patterns.count('BAR')
    output = []
    bars = [i for i, val in enumerate(drum_patterns) if val == 'BAR']
    old_bar = 0;
    i = 0
    for bar in bars:
        if old_bar == 0:
            encoded_drums = drum_patterns[:bar]
        else:
            encoded_drums = drum_patterns[(old_bar + 1):bar]
            # print(encoded_drums)
        old_bar = bar
        try:
            encoded_drums = [ele for ele in encoded_drums if ele not in ['BAR','']]
        except:
            pdb.set_trace()
        output.append(encoded_drums)
    output = list(itertools.chain(*output))
    output = [ele for ele in output if ele not in ['']]
    with open(filename[0:-4]+'noBar.txt', "w") as f:
        for word_idx, word in enumerate(output):
            f.write(word + ' ')
    if returnFilename:
        return filename
    else:
        return

def repeatBar(filename= None,repeatbar = None,returnFilename = False):

    corpus_rep = []
    f = open(filename, 'r')
    new_filename = filename[0:-4] + '_rpBar.txt'
    sentence = f.readline()
    drum_patterns = sentence.split(' ')
    first_bar_idx = drum_patterns.index('BAR')
    n_loops = drum_patterns.count('BAR')
    bars = [i for i, val in enumerate(drum_patterns) if val == 'BAR']
    old_bar = 0;
    i = 0
    for bar in bars:
        if old_bar == 0:
            encoded_drums = drum_patterns[:bar + 1]
            for repeats in range(0, repeatbar):
                corpus_rep.append(encoded_drums)

        else:
            encoded_drums = drum_patterns[(old_bar + 1):bar + 1]
            for repeats in range(0, repeatbar):
                corpus_rep.append(encoded_drums)

        old_bar = bar
    corpus_rep = list(itertools.chain(*corpus_rep))
    corpus_rep = [ele for ele in corpus_rep if ele not in ['']]
    with open(new_filename, "w") as f:
        for word_idx, word in enumerate(corpus_rep):
            f.write(word + ' ')
    if returnFilename:
        return new_filename
    else:
        return

def append_corpus_inDir(filename = None):

    dir = askdirectory()
    allfiles = os.listdir(dir)
    f2 = open(filename, "w")
    for i in allfiles:
        f = open(dir+'/'+i,'r')
        for line in f.readlines():
            f2.write(line)
        f.close()
    f2.close()

def separate_words(filename):
    temp = []
    with open(filename, "r") as f:

        for line in f.readlines():
            for word in line.split():
                if len(word) > 6:
                    temp.append(word[0:3])
                    temp.append(word[3:9])
                else:
                    temp.append(word)
    f.close()


    with open(filename, "w") as f:
        for word in temp:
            f.write(word+' ')

def rtbpat_to_notes(rtbpattern):
    trackNames = []
    event_per_bar = 16
    PPQ = 96
    min_ppq = PPQ / (event_per_bar / 4)
    # mid = MidiFile()
    # mid.ticks_per_beat = PPQ

    step_idx = 0
    note_list = Note_List()

    for i, track in enumerate(rtbpattern):
        trackNames.append(track[1])
        for steps in enumerate(track[1:17]):
            c_tick_here = step_idx * min_ppq
            for item in range(0,len(steps[1][:])):
                nnota = Note(steps[1][item], c_tick_here)
                note_list.add_note(nnota)
            step_idx += 1
    return note_list



def rtbpatSeq_to_notes(rtbpattern):
    trackNames = []
    event_per_bar = 16
    PPQ = 96
    min_ppq = PPQ / (event_per_bar / 4)
    # mid = MidiFile()
    # mid.ticks_per_beat = PPQ

    step_idx = 0
    note_list = Note_List()

    for i, el in enumerate(rtbpattern):

        c_tick_here = i * min_ppq
        for item in range(0,len(el)):
            nnota = Note(el[item], c_tick_here)
            note_list.add_note(nnota)
    return note_list


def conv_text_to_midi(filename=None):


    f = open(filename, 'r')

    sentence = f.readline()
    drum_patterns = sentence.split(' ')
    drum_patterns = [ele for ele in drum_patterns if ele not in ['']]


    bars = [i for i, val in enumerate(drum_patterns) if val == 'BAR']
    old_bar = 0;
    i = 0
    for bar in bars:
        if old_bar == 0:
            encoded_drums = drum_patterns[:bar]
        else:
            encoded_drums = drum_patterns[(old_bar+1):bar]
            #print(encoded_drums)
        old_bar = bar
        try:
            encoded_drums = [ele for ele in encoded_drums if ele not in ['BAR','']]
        except:
            pdb.set_trace()

        # prepare output
        note_list = Note_List()
        # ??
        PPQ = 96
        min_ppq = PPQ / (event_per_bar / 4)
        mid = MidiFile()
        mid.ticks_per_beat = PPQ
        mid.type = 0
        track = MidiTrack()
        mid.tracks.append(track)

        if filename is not None:
            track.append(MetaMessage("track_name", name=os.path.split(filename)[1], time=int(0)))
            track.append(MetaMessage("set_tempo", tempo=480000, time=int(0)))
            track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=int(0)))
        sort_events = []

        vel = 100
        duration = min_ppq*((min_ppq-13)/min_ppq)

        note_list = text_to_notes(encoded_drums, note_list=note_list)
        note_list.quantise(min_ppq)

        for note_idx, note in enumerate(note_list.notes[:-1]):
            sort_events.append([note.pitch, 1, note.c_tick])
            sort_events.append([note.pitch, 0, note.c_tick+duration])

        sort_events.sort(key=lambda tup: tup[2])

        lapso = 0
        for evt in sort_events:
            if evt[1] == 1:
                track.append(Message('note_on', note=evt[0], velocity=vel, time=int((evt[2] - lapso))))
                lapso = evt[2]
            elif evt[1] == 0:
                track.append(Message('note_off', note=evt[0], velocity=0, time=int((evt[2] - lapso))))
                lapso = evt[2]

        if filename is not None:
            track.append(MetaMessage('end_of_track', time=(int(0))))
            mid.save(filename[:-4]+'_'+str(i)+'.mid')
            print('File saved as %s'%(filename[:-4]+'_'+str(i)+'.mid'))
        i += 1
    return



def tobinlist(patterns):
    binpat = []
    k = 36 #kick
    s = 38 #snare
    t = 47 #MidTom
    h = 42 #ClosedHat
    for item in patterns:
        for steps in item:
            bin = 0
            temp = []
            bits = 0
            if isinstance(steps,str):
                binpat.append(steps)

            else:
                if str(t) in str(steps):
                    bin = bin + 1000

                if str(h) in str(steps):
                    bin = int(bin) + 100

                if str(s) in str(steps):
                    bin = int(bin) + 10

                if str(k) in str(steps):
                    bin = int(bin) + 1

                int2st = str(bin).zfill(4)
                #bits = bitarray(int2st)
                #temp.append(bits)
                binpat.append(int2st)

    return binpat

def reduce_drums(mifi,dirr=None):

    mid3 = mido.MidiFile(ticks_per_beat=mifi.ticks_per_beat)
    new_track = mido.MidiTrack()
    mid3.type = 0

    drum_conversion = {35: 36,  # acoustic bass drum -> bass drum (36)
                       37: 38, 40: 38, 39: 38, 65: 38, 66: 38,  # All snares and Timbale to --> Acous. Snare
                       41: 47, 43: 47, 45: 47, 48: 47, 50: 47, 60: 47, 60: 47, 61: 47, 62: 47, 63: 47, 64: 47,# All toms --> Low Mid-Tom
                       42: 42, 44: 42, 49: 42, 51: 42, 52: 42, 54: 42, 55: 42, 57: 42, 59: 42, 46: 42,
                       # All HH / all Cymbal --> To Closed HH
                       }
    # k, sn,cHH,oHH,LFtom,ltm,htm,Rde,Crash
    allowed_pitch = [36, 38, 42, 47]  # 46: open HH


    for i, track in enumerate(mifi.tracks):  # note_on channel=0 note=36 velocity=100 time=0
        print('Track {}: {}'.format(i, track.name))
        for msg in track:
            if str(msg)[0] != '<':

                chan = int(str(msg).split()[1][8:])
                nota = int(str(msg).split()[2][5:7])
                vel = int(str(msg).split()[3][9:12])
                tem = int(str(msg).split()[4][5:7])
                if nota in drum_conversion:
                    nota = drum_conversion[nota]
                if nota not in allowed_pitch:
                    nota = 0
                mess = mido.Message(str(msg).split()[0], channel=chan, note=nota, velocity=vel, time=tem)
                # print(mess)
                new_track.append(mess)
            else:
                new_track.append(msg)
                # print(msg)
    mid3.tracks.append(new_track)

    if not dirr == None:
        filename = 'reduced_' + str(mifi).split(".mid")[0][len(str(mifi).split(".mid")[0]) - 1:len(str(mifi).split(".mid")[0])] + '.mid'
        foldername = str(dirr).split("/")
        foldername = foldername[len(foldername) - 1] +  '/'
        lastname = '../../mod_midis/reduced_' + foldername + filename
        mid3.save(lastname)
        print('File has been saved in ' + lastname)

    else:
        return mid3

def play_midi(music_file):
    # stream music with mixer.music module in blocking manner
    # this will stream the sound from disk while playing

    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print("Music file %s loaded!" % music_file)
    except pygame.error:
        print("File %s not found! (%s)" % music_file, pygame.get_error())
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)


