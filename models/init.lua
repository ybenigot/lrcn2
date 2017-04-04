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

gpu=false

if gpu then
      require 'cunn'
      require 'cudnn'
      file_name = 'gpu_model_best.t7'
      tensor_name = 'torch.CudaTensor'
else
      file_name = 'model_cpu.t7'
      tensor_name = 'torch.FloatTensor'
end   

local M = {}

function M.setup()
   local model

   print('=> Creating model from file resnet.lua')
   model = require('models/resnet')

   --- set model file weights name
   local modelPath = paths.concat('../models', file_name)
   
   assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
   print('=> Resuming model from ' .. modelPath)
   model = torch.load(modelPath):type(tensor_name)
      
   model.__memoryOptimized = nil

   local softMaxLayer = nn.SoftMax()

   -- add Softmax layer
   model:add(softMaxLayer)

   -- Evaluate mode
   model:evaluate()

   return model
end

return M
