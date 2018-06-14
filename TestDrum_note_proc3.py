from tkinter.filedialog import askdirectory
from Datasets import import_midis_dir, process_midi, reduce_drums, parse_midi_directory,tobinlist
from midi_playback import play_midi
import rhythmtoolbox as rtb
import os
import tensorflow as tf
import pymidifile
import mido
import note_proc as dnp
from midi_drums_utils import conv_text_to_midi, text_to_notes



PPQ = 96  # Pulse per quater note
event_per_bar = 16  # to quantise / numero timesteps
min_ppq = PPQ / (event_per_bar / 4)

"""
def text_to_notes(encoded_drums, note_list = None):

    if note_list == None:
        note_list = dnp.Note_List()
    for word_idx, word in enumerate(encoded_drums):
        c_tick_here = word_idx*dnp.min_ppq
        for pitch_idx, pitch in enumerate(dnp.allowed_pitch):

            if word[pitch_idx+2] == '1':
                n_nota = dnp.Note(pitch,c_tick_here)
                note_list.add_note(n_nota)
    return note_list

"""




dir  = askdirectory()
allfiles = os.listdir(dir)

listFiles = []
for file in allfiles:
    lista = dnp.Note_List()
    elapsed = 0
    mid = mido.MidiFile(dir+'/'+file)
    if mid.ticks_per_beat != 96:
        min_ppq = mid.ticks_per_beat/4
# Con esto ejecutamos la codificación midi
    for tracks in mid.tracks:
        for msg in tracks:
            elapsed += msg.time
            if str(msg)[0] != '<':
                if msg.type == 'note_on':
                    print(msg)
                    nota = dnp.Note(msg.note,elapsed)
                    lista.add_note(nota)
    lista.quantise(min_ppq)
    lista.simplify_drums()
    strbin = lista.return_as_text()
    strbin = strbin.split(' ')
    listFiles.append(strbin)




### Esta parte sirve para los midis con otra resolucion al recuantizarlos queda un BAR todo ceros  y da error
#   ESTO elimina ese bar
if mid.ticks_per_beat != 96:
    for files in listFiles:
        Ocount = 0
        indx = 0
        for el in files:
            if el == '0b0000':
                Ocount +=1
            if el =='BAR':
                indx += 1
                if Ocount == event_per_bar:
                    files[:]=files[0:(indx*17-1)-event_per_bar]
                    Ocount = 0
                else:
                    Ocount = 0


for i, files in enumerate(listFiles):
    with open('Booka_Shade_sel'+str(i)+'_.txt','w') as f:
        for word in files:
            f.write(word+' ')
    f.close()



"""
count = 0
for files in listFiles:
  if not count:
      count += 1
      with open("scruff.txt", "w") as f:
          for word_idx, word in enumerate(files):
              f.write(word + ' ')
          f.close()
  else:
      with open("scruff.txt", "a") as f:
          for word_idx, word in enumerate(files):
              f.write(word + ' ')
          f.close()



"""
