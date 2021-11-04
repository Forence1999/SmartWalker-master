#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
    loop thread to run ssl
"""
from scipy import stats
import numpy as np
from pyaudio import PyAudio, paInt16
from SoundSourceLocalization.ssl_setup import *
from SoundSourceLocalization.ssl_feature_extractor import FeatureExtractor
# from SoundSourceLocalization.ssl_actor_critic import Actor, Critic
from SoundSourceLocalization.ssl_map import Map
from SoundSourceLocalization.ssl_audio_processor import *
from SoundSourceLocalization.ssl_turning import SSLturning
from SoundSourceLocalization.kws_detector import KwsDetector
import time
import sys
import os
import threading
import random
from lib import utils
from lib.utils import standard_normalizaion, add_prefix_and_suffix_4_basename
from lib.audiolib import normalize_single_channel_to_target_level, audio_segmenter_4_numpy, \
    audio_energy_ratio_over_threshold, audio_energy_over_threshold, audiowrite, audioread
import ns_enhance_onnx

from SoundSourceLocalization.ssl_DOA_model import DOA

pwd = os.path.abspath(os.path.abspath(__file__))
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..")
sys.path.append(father_path)
import Driver.ControlOdometryDriver as CD


class SSL:
    def __init__(self, denoise=True, seg_len='1s', debug=False):
        print('-' * 20 + 'init SSL class' + '-' * 20)
        # self.KWS = KwsDetector(CHUNK, RECORD_DEVICE_NAME, RECORD_WIDTH, CHANNELS,
        #                        SAMPLE_RATE, FORMAT, KWS_WAVE_PATH, KWS_MODEL_PATH, KWS_LABEL_PATH)
        self.micro_mapping = np.array(range(CHANNELS), dtype=np.int)
        self.denoise = denoise
        self.device_index = self.__get_device_index__()
        self.frames = []
        segment_para_set = {
            '32ms' : {
                'name'     : '32ms',
                'time_len' : 32 / 1000,
                'threshold': 100,
                'stepsize' : 0.5
            },
            '50ms' : {
                'name'     : '50ms',
                'time_len' : 50 / 1000,
                'threshold': 100,
                'stepsize' : 0.5
            },
            '64ms' : {
                'name'     : '64ms',
                'time_len' : 64 / 1000,
                'threshold': 100,
                'stepsize' : 0.5
            },
            '128ms': {
                'name'     : '128ms',
                'time_len' : 128 / 1000,
                'threshold': 200,  # 100?
                'stepsize' : 0.5
            },
            '256ms': {
                'name'     : '256ms',
                'time_len' : 256. / 1000,
                'threshold': 400,
                'stepsize' : 256. / 1000/2
            },
            '1s'   : {
                'name'     : '1s',
                'time_len' : 1.,
                'threshold': 800,
                'stepsize' : 0.5
            },
        }
        self.fs = SAMPLE_RATE
        self.num_gcc_bin = 128
        self.num_mel_bin = 128
        self.seg_para = segment_para_set[seg_len]
        self.fft_len = utils.next_greater_power_of_2(self.seg_para['time_len'] * self.fs)
        self.debug = debug
        self.save_dir_name = ''
        ref_audio, _ = audioread('../resource/wav/reference_wav.wav')
        self.ref_audio = normalize_single_channel_to_target_level(ref_audio)
        self.ref_audio_threshold = (self.ref_audio ** 2).sum() / len(self.ref_audio) / 500
        print('Loading denoising model...\n')
        self.denoise_model, _ = ns_enhance_onnx.load_onnx_model()
        print('Loading DOA model...\n')
        self.doa = DOA(model_dir=os.path.abspath('./model/EEGNet/ckpt'), fft_len=self.fft_len,
                       num_gcc_bin=self.num_gcc_bin, num_mel_bin=self.num_mel_bin, fs=self.fs, )
    
    def __get_device_index__(self):
        device_index = -1
        
        # scan to get usb device
        p = PyAudio()
        print('num_device:', p.get_device_count())
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            device_name = info.get("name")
            print("device_name: ", device_name)
            
            # find mic usb device
            if device_name.find(RECORD_DEVICE_NAME) != -1:
                device_index = index
                break
        
        if device_index != -1:
            print('-' * 20 + 'Find the device' + '-' * 20 + '\n', p.get_device_info_by_index(device_index), '\n')
            del p
        else:
            print('-' * 20 + 'Cannot find the device' + '-' * 20 + '\n')
            exit()
        
        return device_index
    
    def savewav_from_frames(self, filename, frames=None):
        if frames is None:
            frames = self.frames
        
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(RECORD_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def save_multi_channel_audio(self, des_dir, audio, fs=SAMPLE_RATE, norm=True, ):
        for i in range(len(audio)):
            file_path = os.path.join(des_dir, 'test_mic%d.wav' % i)
            audiowrite(file_path, audio[i], sample_rate=fs, norm=norm, target_level=-25, clipping_threshold=0.99)
    
    def read_multi_channel_audio(self, dir_path, num_channel=CHANNELS):
        audio = []
        for i in range(num_channel):
            file_path = os.path.join(dir_path, 'test_mic%d.wav' % i)
            audio_i, _ = audioread(file_path, )
            audio.append(audio_i)
        return np.array(audio)
    
    def read_and_split_channels_from_file(self, filepath):
        f = wave.open(filepath)
        params = f.getparams()
        num_channel, sample_width, fs, num_frame = params[:4]
        str_data = f.readframes(num_frame)
        f.close()
        audio = np.frombuffer(str_data, dtype=np.short)
        audio = np.reshape(audio, (-1, 4)).T
        
        return audio
    
    def split_channels_from_frames(self, frames=None, num_channel=CHANNELS, mapping_flag=True):
        if frames is None:
            frames = self.frames
        audio = np.frombuffer(b''.join(frames), dtype=np.short)
        audio = np.reshape(audio, (-1, num_channel)).T
        if mapping_flag:
            audio = audio[self.micro_mapping]
        return audio
    
    def monitor_from_4mics(self, record_seconds=RECORD_SECONDS):
        # print('-' * 20 + "start monitoring ...")
        p = PyAudio()
        stream = p.open(format=p.get_format_from_width(RECORD_WIDTH),
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=self.device_index)
        # 16 data
        frames = []
        
        for i in range(int(SAMPLE_RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        # print('-' * 20 + "End monitoring ...\n")
        
        return frames
    
    def monitor_audio_and_return_amplitude_ratio(self, mapping_flag):
        frames = self.monitor_from_4mics(record_seconds=1)
        audio = self.split_channels_from_frames(frames=frames, num_channel=CHANNELS, mapping_flag=mapping_flag)
        amp2_sum = np.sum(standard_normalizaion(audio) ** 2, axis=1).reshape(-1)
        amp2_ratio = amp2_sum / amp2_sum.sum()
        
        return amp2_ratio
    
    def init_micro_mapping(self, ):
        print('Please tap each microphone clockwise from the upper left corner ~ ')
        mapping = [None, ] * 4
        while True:
            for i in range(CHANNELS):
                while True:
                    ratio = self.monitor_audio_and_return_amplitude_ratio(mapping_flag=False)
                    idx = np.where(ratio > 0.5)[0]
                    if len(idx) == 1 and (idx[0] not in mapping):
                        mapping[i] = idx[0]
                        print(' '.join(['Logical channel', str(i), 'has been set as physical channel', str(mapping[i]),
                                        'Amplitude**2 ratio: ', str(ratio)]))
                        break
            print('Final mapping: ')
            print('Logical channel: ', list(range(CHANNELS)))
            print('Physical channel: ', mapping)
            break
            
            confirm_info = input('Confirm or Reset the mapping? Press [y]/n :')
            if confirm_info in ['y', '', 'yes', 'Yes']:
                break
            else:
                print('The system will reset the mapping')
                continue
        self.micro_mapping = np.array(mapping)
    
    def save_wav(self, filepath):
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.SAMPLING_RATE)
        wf.writeframes(np.array(self.Voice_String).tostring())
        # wf.writeframes(self.Voice_String.decode())
        wf.close()
    
    def drop_audio_per_seg_point(self, signal_segment, ):
        '''
        two standards:
        1. audio_energy_ratio
        2. audio_energy_over_threshold
        '''
        signal_mean = signal_segment.mean(axis=0)
        
        return not (audio_energy_over_threshold(signal_mean, threshold=self.ref_audio_threshold, ) and
                    audio_energy_ratio_over_threshold(signal_mean, fs=SAMPLE_RATE,
                                                      threshold=self.seg_para['threshold'], ))
    
    def save_continuous_True(self, ini_list, num=3):  # todo
        pass
    
    def drop_audio_clips(self, signal_segments, ):
        # print('Number of segments before dropping: ', len(signal_segments))
        audio_segments = []
        drop_flag = []
        for i in range(len(signal_segments)):
            drop_flag.append(self.drop_audio_per_seg_point(signal_segments[i]))
            if not drop_flag[-1]:
                audio_segments.append(signal_segments[i])
            else:
                continue
                # audio_segments.append([])
        # print('Number of segments after dropping: ', len(audio_segments))
        
        return np.array(audio_segments), drop_flag
    
    def concat_ref_audio(self, audios, ref_audio):
        res_audio = []
        for i in audios:
            res_audio.append(np.concatenate((ref_audio, i)))
        
        return np.array(res_audio)
    
    def del_ref_audio(self, audios, ref_audio):
        audios = np.array(audios)
        length = len(ref_audio)
        return audios[:, length:]
    
    def norm_batch_audio_to_target_level(self, audio_batch):
        res_audio = []
        for audio_channels in audio_batch:
            norm_audio_channels = []
            for audio in audio_channels:
                norm_audio_channels.append(normalize_single_channel_to_target_level(audio))
            res_audio.append(norm_audio_channels)
        
        return np.array(res_audio)
    
    def denoise_batch_audio(self, audio_batch):
        res_audio = []
        for audio_channels in audio_batch:
            denoised_channels = []
            for audio in audio_channels:
                denoised_channels.append(
                    ns_enhance_onnx.denoise_nsnet2(audio=audio, fs=SAMPLE_RATE, model=self.denoise_model, ))
            res_audio.append(denoised_channels)
        
        return np.array(res_audio)
    
    def preprocess_ini_signal(self, ini_signals):
        # todo how to denoise when nobody is talking
        ini_signals = np.array(ini_signals, dtype=np.float64)
        segs = np.array([audio_segmenter_4_numpy(signal, fs=self.fs, segment_len=self.seg_para['time_len'],
                                                 stepsize=self.seg_para['stepsize'], window='hann', padding=False,
                                                 pow_2=True) for signal in ini_signals]).transpose(1, 0, 2)
        # norm_segs = segs
        norm_segs = self.norm_batch_audio_to_target_level(segs)
        
        # norm_signals = self.concat_ref_audio(norm_signals, self.ref_audio)
        # denoised_norm_signals = self.del_ref_audio(denoised_norm_signals, self.ref_audio)
        
        denoised_norm_segs = self.denoise_batch_audio(audio_batch=norm_segs)
        
        drop_denoised_norm_segs, _ = self.drop_audio_clips(signal_segments=denoised_norm_segs)
        final_segments = self.norm_batch_audio_to_target_level(drop_denoised_norm_segs)
        
        return final_segments, None
    
    def loop(self, event, control, source='test'):
        self.init_micro_mapping()
        
        # initialize models
        # map = Map()
        # gccGenerator = GccGenerator()
        # actor = Actor(GCC_BIAS, ACTION_SPACE, lr=0.004)
        # critic = Critic(GCC_BIAS, ACTION_SPACE, lr=0.003, gamma=0.95)
        # actor.load_trained_model(MODEL_PATH)
        
        # init at the first step
        # state_last = None
        # action_last = None
        # direction_last = None
        
        # steps
        while True:
            event.wait()
            # print('num_saved_sig: ', int(num_saved_sig))
            # map.print_walker_status()
            # map.detect_which_region()
            
            # final_file = None
            
            # Record
            # # todo, congest here for kws
            # if num_saved_sig == 0:
            #     print("congest in KWS ...")
            #     self.KWS.slide_win_loop()
            #     wakeup_wav = self.KWS.RANDOM_PREFIX + "win.wav"
            #
            #     denoised_sig_fname = str(num_saved_sig) + "_de.wav"
            #
            #     de_noise(os.path.join(self.KWS.WAV_PATH, wakeup_wav),
            #              os.path.join(self.KWS.WAV_PATH, denoised_sig_fname))
            #
            #     if self.denoise is False:
            #         final_file = wakeup_wav
            #     else:
            #         final_file = denoised_sig_fname
            #
            # else:
            #     # active detection
            #     print("start monitoring ... ")
            #     while True:
            #         event.wait()
            #         # print("start monitoring ... ")
            #         frames = self.monitor_from_4mics()
            #
            #         # store the signal
            #         file_name = os.path.join(WAV_PATH, str(num_saved_sig) + ".wav")
            #         self.savewav(file_name, frames)
            #
            #         # de-noise the signal into new file, then VAD and split
            #         denoised_sig_fname = str(num_saved_sig) + '_denoised.wav'
            #         de_noise(os.path.join(WAV_PATH, ini_sig_fname), os.path.join(WAV_PATH, denoised_sig_fname))
            #
            #         # if exceed, break, split to process, then action. After action done, begin monitor
            #
            #         if self.de is False:
            #             final_file = ini_sig_fname
            #         else:
            #             final_file = denoised_sig_fname
            #
            #         if judge_active(os.path.join(WAV_PATH, final_file)):
            #             print("Detected ... ")
            #             break
            #
            # Split
            
            # produce action
            """
                use four mic file to be input to produce action
            """
            if self.debug:
                self.save_dir_name = 'self_collected'
                ini_dir = os.path.join(WAV_PATH, self.save_dir_name, 'ini_signal')
                ini_signals = self.read_multi_channel_audio(ini_dir, num_channel=CHANNELS)
            else:
                # self.save_dir_name = 'test'
                frames = self.monitor_from_4mics()
                ini_signals = self.split_channels_from_frames(frames=frames, num_channel=CHANNELS, mapping_flag=True)
                # ini_dir = os.path.join(WAV_PATH, self.save_dir_name, 'ini_signal')
                # self.save_multi_channel_audio(ini_dir, ini_signals, fs=SAMPLE_RATE, norm=False, )
            
            audio_segments, drop_flag = self.preprocess_ini_signal(ini_signals)
            # print('Number of preprocessed audio segments: ', len(audio_segments))
            direction = None
            if len(audio_segments) > 0:
                
                gcc_feature_batch = self.doa.extract_gcc_phat_4_batch(audio_segments)
                # length=len(gcc_feature_batch)
                gcc_feature = np.mean(gcc_feature_batch, axis=0)[np.newaxis,]
                direction_prob, direction_cate, = self.doa.predict(gcc_feature)
                # print(direction_prob)
                print('Pridict value: ', direction_cate)
                # direction = stats.mode(direction_cate)[0][0] * 45
                direction = (360-(int(direction_cate) * 45  - 90)) % 360
                print("producing action ...\n", 'Direction', direction)
                SSLturning(control, direction)
                control.speed = STEP_SIZE / FORWARD_SECONDS
                control.radius = 0
                control.omega = 0
                time.sleep(FORWARD_SECONDS)
                control.speed = 0
                print("movement done.")
            #
            # print('Wait ~ ')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    ssl = SSL(denoise=True, seg_len='256ms', debug=False)
    cd = CD.ControlDriver(left_right=0)
    # cd = ''
    temp = threading.Event()
    temp.set()
    p2 = threading.Thread(target=cd.control_part, args=())
    p1 = threading.Thread(target=ssl.loop, args=(temp, cd,))
    
    p2.start()
    p1.start()
