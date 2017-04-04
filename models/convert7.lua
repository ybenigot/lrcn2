--[[
A quick script for converting GPU checkpoints to CPU checkpoints.
CPU checkpoints are not saved by the training script automatically
because of Torch cloning limitations. In particular, it is not
possible to clone a GPU model on CPU, something like :clone():float()
with a single call, without needing extra memory on the GPU. If this
existed then it would be possible to do this inside the training
script without worrying about blowing up the memory.
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn' -- only needed if the loaded model used cudnn as backend. otherwise can be commented out
-- local imports
require 'misc.LanguageModel'

-- th convert7.lua gpu_model_best.t7

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert a GPU checkpoint to CPU checkpoint.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','GPU model checkpoint to convert')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed

print ("open model : "..opt.model)
checkpoint = torch.load(opt.model)
protos = checkpoint.protos

local opt = checkpoint.opt
print('opt: ')
print(opt)
print('val_losses: ')
print(checkpoint.val_losses)
print('protos: ')
print(protos)
print('checkpoint : ')
print (checkpoint)



local savefile = 'model' .. '_cpu.t7' -- append "cpu.t7" to filename
checkpoint=checkpoint:float()
checkpoint=cudnn.convert(checkpoint,nn) 
torch.save(savefile, checkpoint)
print('saved ' .. savefile)
