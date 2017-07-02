--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--
-- adapted for resnet use, preidction setup

require 'nn'
require 'torch'

if GPU then
      require 'cunn'
      require 'cudnn'
      MODEL_FILE = 'resnet_gpu_model_best.t7'
      TENSOR_NAME = 'torch.CudaTensor'
else
      MODEL_FILE = 'resnet_cpu_model.t7'
      TENSOR_NAME = 'torch.FloatTensor'
end   

local MODEL_DIR=nil
if MACHINE == 'mac' then
   MODEL_DIR='/Users/yves/Dropbox/torch/lrcn2/parameters'
else
   MODEL_DIR='/home/yves/Dropbox/torch/lrcn2/parameters'
end

print(MODEL_DIR,MODEL_FILE)

local M = {}

function M.setup(predict)
   local model

   print('=> Creating model from file resnet.lua')
   model = require('models/resnet')

   --- set model file weights name
   local modelPath = paths.concat(MODEL_DIR, MODEL_FILE)
   assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
   print('=> Resuming model from ' .. modelPath)
   model = torch.load(modelPath):type(TENSOR_NAME)
      
   model.__memoryOptimized = nil

   local softMaxLayer = nn.SoftMax()

   if predict then
     -- add Softmax layer
     print('add softmax to resnet for prediction')
     model:add(softMaxLayer)
   end

   -- Evaluate mode
   model:evaluate()

   -- compute output layer size
   modules=model.modules

   --print('model topology ',#modules)
   --print(modules)
   --output_size = modules[#modules].output.size()

   return model
end

return M
