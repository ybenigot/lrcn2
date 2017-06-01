#!/usr/bin/env wsapi.cgi

-- autoconfiguration based on hostname
local hostname=io.popen('hostname'):read()
if string.sub(hostname,1,7) == 'yvesMBP' then
  MACHINE='mac'
elseif hostname == 'pcybldlc' then
  MACHINE='pc'
else  
  MACHINE='pc'
end
print("Machine : ",MACHINE)

-- we do not use GPU for prediction
GPU=false
    function to_cuda(obj) 
      return obj
    end 

local orbit = require "orbit"

require 'torch'
require 'paths'
require 'nn'
require 'image'
local cnn_model  = require 'models/init'
local lstm_model = require 'models/caption'
local GloVe = require 'glove/glove'

-- general network topology parameters
local MAX_SENTENCE_LENGTH = 20
local CNN_OUTPUT_SIZE = 1000
local SIZE=224 -- size of neural net input images 

module("predict", package.seeall, orbit.new)

  -- init function for loading the model
  local function ml_init()
    -- configure
    print("configure torch")
    package.path = '../?.lua;' .. package.path
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.setnumthreads(1)
    
    -- now load the models
    print("load cnn model ...")
    cnn, LSTM_INPUT_SIZE = cnn_model.setup(false)
    cnn = nn.Sequencer(cnn)
    print("... model loaded")
    print(model)

    print("create and init lstm model...")
    lstm = lstm_model.createCaptionModel(LSTM_INPUT_SIZE, GloVe:vector_size())
    lstm = nn.Sequencer(lstm)
    params = torch.load('lstm-params-cpu-1.t7')
    parameters = lstm:getParameters()
    parameters = params:clone()
    print("...created lstm")
  end

print("using ".._VERSION)

ml_init()

local image_name ='temp.jpg'

local counter = 0


-- controllers

function index(web)
  return render_index('',false)
end

-- process the image and do the prediction work using the model 
local function prediction(web)

  -- get image POSTed as post parameter
  print("process post")
  parameter = web.POST["file"]
  data = parameter.contents

  -- write image to local file, useful for debug
  local f = assert(io.open(image_name, 'wb')) 
  f:write(data)
  f:close()
  
  -- resize image to neural network input size
  image_size = image.getSize(image_name)
  print('image dimensions : '..image_size[1]..'x'..image_size[2]..'x'..image_size[3])
  img = image.load(image_name)
  img = image.scale(img,SIZE,SIZE)
  input = img:view(1,3,SIZE,SIZE)
  
  -- overwrite image with rescaled image
  image.save(image_name, img) 

  -- apply the cnn model loaded by ml_init()
  local output = cnn:forward(input):float()

  -- make an lstm input by repeating cnn output
  local BatchInputTable = {}
  for s = 1, MAX_SENTENCE_LENGTH do -- fixme define stop criterium ?
    BatchInputTable[s] = torch.reshape(output,CNN_OUTPUT_SIZE)
  end

  -- finally, use lstm model to compute a sequence of predicted word vectors
  local words = lstm:forward(BatchInputTable)
  local sentence
  for i, word in pairs(words) do
    local result = GloVe:distance(word)
    local returnwords = result[2]
    if sentence == nil then
      sentence = GloVe:vec2word(word)
    else  
      sentence = sentence..' '..GloVe:vec2word(word)
    end  
  end  
  return render_index(sentence,true)
end

predict:dispatch_get(index, "/", "/index")
predict:dispatch_post(prediction, "/do_predict")
predict:dispatch_static('/'..image_name)

-- views

function render_index(label, display_image)

  counter = counter + 1

  local header    = '<!doctype html><html><body><h3>Predict '..counter..'</h3><br/>'
  local image_tag = display_image and ('<img src="'..image_name..'"/><br/>') or '--'
  local result    = '<br/>'..label..'<br/>'
  local form      = [[<form action="/do_predict" method="post" enctype="multipart/form-data">
                         <input type=file name="file"><br/>
                         <input type="submit">
                   </form>]]
  local footer = '</body></html>'

  print('response '..counter..' done')

  return header..image_tag..result..form..footer

end


return _M

