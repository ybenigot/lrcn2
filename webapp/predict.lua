#!/usr/bin/env wsapi.cgi

-- use webapp/serve.sh to start this

require 'torch'
require 'paths'
require 'nn'
require 'rnn'
require 'nngraph'
require 'image'

nn.FastLSTM.usenngraph = true -- faster
nn.FastLSTM.bn = true -- normalize hidden values

-- autoconfiguration based on hostname
local hostname=io.popen('hostname'):read()
if string.sub(hostname,1,7) == 'yvesMBP' then
  MACHINE='mac'
  --LSTM_FILE  = '/Users/yves/Documents/lrcn2/parameters/lstm-float-2106.t7'
  LSTM_FILE ='/Users/yves/Documents/lrcn2/parameters/lstm-params-2-float.t7'
  LSTM1_FILE = '/Users/yves/Documents/lrcn2/parameters/lstm1-float.t7'
  LSTM2_FILE = '/Users/yves/Documents/lrcn2/parameters/lstm2-float.t7'
elseif hostname == 'pcybldlc' then
  MACHINE='pc'
  LSTM_FILE  = '/home/yves/save/lrcn2/parameters/lstm-params-3006-float.t7'
  LSTM1_FILE = '/home/yves/save/lrcn2/parameters/lstm1-params.t7'
  LSTM2_FILE = '/home/yves/save/lrcn2/parameters/lstm2-params.t7'
else  
  MACHINE='pc'
end
print("Machine : ",MACHINE)

-- we do not use GPU for prediction
GPU=false

function to_cuda(obj) 
  return obj
end 

local cnn_model  = require 'models/init'
local GloVe = require 'glove/glove'
require 'hyperparams'
local orbit = require "orbit"

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
    local embedding_size = GloVe:vectorSize()
    local dict_size = GloVe:dictionarySize()

    local load_model = true
    if load_model then   
      lstm = torch.load(LSTM_FILE) -- parameters/lstm-params-final.t7
      lstm.module.module:add(nn.LogSoftMax()) -- add sofmax under the sequuencer/recurrent 
    else -- load individual lstm layers
      local lstm_model = require 'models/caption'
      lstm = lstm_model.createCaptionModel(CNN_OUTPUT_SIZE, dict_size, embedding_size, GloVe:encoding(), 
                                         MAX_SENTENCE_LENGTH, true, LSTM1_FILE, LSTM2_FILE,"binary")
    end

    print("...created lstm", lstm)
    encode_layer = nn.Linear(dict_size, embedding_size)
    encode_layer.weight[{ {1, embedding_size} , {1, dict_size} }] = GloVe:encoding():transpose(1,2)
    print("...created encoding layer",encode_layer)
  end

print("using ".._VERSION)

ml_init()

local image_name ='temp.jpg'

local counter = 0

-- controllers

function index(web)
  return render_index('',false)
end

local function lstm_prediction(cnn_output)
  -- run the lstm iteratively to compute a sentence
  local sentence

  local previous_output = cnn_output:clone() -- the first input is the image, then the input will be the last word

  local raw_lstm=lstm.module.module

  for s = 1, MAX_SENTENCE_LENGTH do -- no stop criterion yet
    local word_output_batch = raw_lstm:forward(previous_output):clone()

    -- here we have word_output which is a sofmax and convert it to a true one hot vector
    local word_output = GloVe:softMax2oneHot(word_output_batch) 

    -- second compute an embedding to reboot into 'previous word' as the lstm input
    local word_ouput_embedded = encode_layer:forward(word_output):clone()

    -- compute current word
    local token = GloVe:token(word_output)

    print("token",token)
    sentence = sentence == nil and token  or  sentence..' '..token 
    previous_output = word_ouput_embedded -- loop back output word to next input
  end

  lstm:forget() -- necessary otherwise the system works only for the first prediction

  return sentence
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

  print("cnn output has been computed")

  local cnn_output = torch.reshape(output, CNN_OUTPUT_SIZE) -- flatten the CNN output

  --torch.save ('/Users/yves/Documents/lrcn2/parameters/cnn_output.t7',cnn_output) -- for tests

  local sentence = lstm_prediction(cnn_output) 

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

--[[
  local input_sequence={}

  for s = 1, MAX_SENTENCE_LENGTH do
    input_sequence[s] = torch.zeros(1, CNN_OUTPUT_SIZE)
    input_sequence[s][1] = cnn_output:clone()
  end  

  lstm_output = lstm:forward(input_sequence)

  for s = 1, MAX_SENTENCE_LENGTH do
    print("lstm ouput",lstm_output[s][1][3981],lstm_output[s][1][3982],lstm_output[s][1][3983],lstm_output[s][1][3984])
    local word_decoded = GloVe:softMax2oneHot(lstm_output[s][1])
    local token = GloVe:token(word_decoded)
    token = token:gsub("<(%w+)>","%1")
    sentence = sentence == nil and token  or  sentence..' '..token
  end  
]]--

