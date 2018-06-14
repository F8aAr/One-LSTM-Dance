import midi
import pdb
import numpy as np

PPQ = 96  # Pulse per quater note
event_per_bar = 16  # to quantise / numero timesteps
min_ppq = PPQ / (event_per_bar / 4) # min resolution
#
drum_conversion = {35: 36,  # acoustic bass drum -> bass drum (36)
                   37: 38, 40: 38, 39: 38, 65: 38, 66: 38,  # All snares and Timbale to --> Acous. Snare
                   41: 47, 43: 47, 45: 47, 48: 47, 50: 47, 60: 47, 60: 47, 61: 47, 62: 47, 63: 47, 64: 47,
                   # All toms --> Low Mid-Tom
                   42: 42, 44: 42, 49: 42, 51: 42, 52: 42, 54: 42, 55: 42, 57: 42, 59: 42, 46: 42,
                   # All HH / all Cymbal --> To Closed HH
                   }
# k, sn,cHH,oHH,LFtom,ltm,htm,Rde,Crash
allowed_pitch = [47, 42, 38, 36]  # 46: open HH


# Keunwoo's classes Note and Note_list
class Note:
    def __init__(self, pitch, c_tick):
        self.pitch = pitch
        self.c_tick = c_tick  # cumulated_tick of a midi note

    def add_index(self, idx):
        '''index --> 16-th note-based index starts from 0'''
        self.idx = idx


class Note_List():
    def __init__(self):
        ''''''
        self.notes = []
        self.quantised = False
        self.max_idx = None

    def add_note(self, note):
        '''note: instance of Note class'''
        self.notes.append(note)

    def quantise(self, minimum_ppq):
        '''
		e.g. if minimum_ppq=120, quantise by 16-th note.

		'''
        if not self.quantised:
            for note in self.notes:
                note.c_tick = ((note.c_tick + minimum_ppq / 2) / minimum_ppq) * minimum_ppq  # quantise
                note.add_index(note.c_tick / minimum_ppq)


            self.max_idx = note.idx
            if (self.max_idx + 1) % event_per_bar != 0:
                self.max_idx += event_per_bar - (
                            (self.max_idx + 1) % event_per_bar)  # make sure it has a FULL bar at the end.
            self.quantised = True
        else:
            print('Allready Quantised')

        return

    def simplify_drums(self):
        ''' use only allowed pitch - and converted not allowed pitch to the similar in a sense of drums!
		'''

        for note in self.notes:
            if note.pitch in drum_conversion:  # ignore those not included in the key
                note.pitch = drum_conversion[note.pitch]

        self.notes = [note for note in self.notes if note.pitch in allowed_pitch]

        return

    def return_as_text(self):
        ''''''
        length = self.max_idx + 1  # of events in the track.
        event_track = []
        for note_idx in range(int(length)):
            event_track.append(['0'] * len(allowed_pitch))
            #if note_idx > 20000:
                #print(note_idx)

        num_bars = length / event_per_bar  # + ceil(len(event_texts_temp) % _event_per_bar)

        for note in self.notes:
            pitch_here = note.pitch
            note_add_pitch_index = allowed_pitch.index(pitch_here)  # 0-8
            event_track[int(note.idx)][note_add_pitch_index] = '1'
        #print (note.idx, note.c_tick, note_add_pitch_index, ''.join(event_track[note.idx]))
        #pdb.set_trace()

        event_text_temp = ['0b' + ''.join(e) for e in event_track]  # encoding to binary

        event_text = []
        # event_text.append('SONG_BEGIN')
        # event_text.append('BAR')
        for bar_idx in range(int(num_bars)):
            event_from = bar_idx * event_per_bar
            event_to = event_from + event_per_bar
            event_text = event_text + event_text_temp[event_from:event_to]
            event_text.append('BAR')
            #print(bar_idx)
        # event_text.append('SONG_END')

        return ' '.join(event_text)
