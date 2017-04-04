#!/usr/bin/env wsapi.cgi

local orbit = require "orbit"

require 'torch'
require 'paths'
require 'nn'
require 'image'


module("predict", package.seeall, orbit.new)

  -- init function for loading the model
  local function ml_init()
    -- configure
    print("configure torch")
    package.path = '../?.lua;' .. package.path
    local models = require 'models/init'
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.setnumthreads(1)
    -- now load the model
    print("load model ...")
    model = models.setup()
    print("... model loaded")
    print(model)
  end

print("using ".._VERSION)

ml_init()

words={} -- the dictionnary of labels corresponding to an output value

  -- load index to prediction label table
  local function load_words()
    local i=1
    for line in io.lines('words1000.txt') do
      words[i] = line
      i = i + 1
    end  
    print('read '..(i-1)..' lines')
  end

load_words()

local image_name ='temp.jpg'

  -- utility functions

  -- given a predicted value as a number, compute its label using the words table
  local function get_label(i)
    if i<=1000 then
      return words[i]
    else
      return 'unknown'
    end    
  end

local counter = 0

  -- controllers

  function index(web)
    return render_index('',false)
  end

  -- process the image and do the prediction work using the model 
  local function prediction(web)

    local SIZE=224 -- size of neural net input images 

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
    -- do the prediction using the model loaded by ml_init()
    local output = model:forward(input):float()
    local maxs, indices = torch.max(output,2)
    local predicted = indices[1]
    print('predicted class number: '..predicted[1])
    -- display the rescaled input image and the prediction
    return render_index(get_label(predicted[1]),true)
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

