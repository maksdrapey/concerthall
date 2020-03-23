import shutil
import os

dirs_gconv0_0 = [os.path.join('gconv0_dreamedfeats','gconv0_0', dd) for dd in os.listdir(os.path.join('gconv0_dreamedfeats','gconv0_0')) if os.path.isdir(os.path.join('gconv0_dreamedfeats','gconv0_0', dd))]
dirs_gconv0_1 = [os.path.join('gconv0_dreamedfeats','gconv0_1', dd) for dd in os.listdir(os.path.join('gconv0_dreamedfeats','gconv0_1')) if os.path.isdir(os.path.join('gconv0_dreamedfeats','gconv0_1', dd))]
dirs_gconv0_2 = [os.path.join('gconv0_dreamedfeats','gconv0_2', dd) for dd in os.listdir(os.path.join('gconv0_dreamedfeats','gconv0_2')) if os.path.isdir(os.path.join('gconv0_dreamedfeats','gconv0_2', dd))]
dirs_gconv1_0 = [dd for dd in os.listdir(os.getcwd()) if 'gconv1_0' in  dd]
dirs_gconv1_1 = [dd for dd in os.listdir(os.getcwd()) if 'gconv1_1' in  dd]

for dd in dirs_gconv1_1:
    giffiles = [fn for fn in os.listdir(dd) if '_250' in fn]
    shutil.copy(os.path.join(dd,giffiles[0]), os.path.join('gconv1_1', dd.split('/')[-1]+'_iter250.gif'))
