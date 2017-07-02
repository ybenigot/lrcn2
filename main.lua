-- adapted for prediction on LRCN (from Facebook code on resnets)
-- references :
--    Facebook resnet code : https://github.com/facebook/fb.resnet.tensor
--    LRCN : http://jeffdonahue.com/lrcn/
--           https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&cad=rja&uact=8&ved=0ahUKEwiOhd64hLvTAhUB0hoKHfxvAhEQFgg6MAM&url=https%3A%2F%2Farxiv.org%2Fabs%2F1411.4389&usg=AFQjCNFgyGMHQYYPP-QSp4Y0FA0q2gdvDg
--
-- this code can do :
--    LRCN training, with a freezed resnet        use LSTM_TRAIN  = true
--    resnet prediction (for testing purposes)    use CNN_PREDICT = true
--    LRCN prediction                             use LSTM_PREDICT = true

require 'paths'
require 'nn'
require 'rnn'
require 'image'
require 'optim'
require 'io'
require 'os'

require 'hwconfig'

local DataReader = require 'datareader'
local models = require 'models/init'
local lstm_model = require 'models/caption'
local GloVe = require 'glove/glove'

-- processing configuration
local DISPLAY = false       -- dipslay or not a sample of images and predictions
local LOG = true           -- write or not image names and predictions
local CNN_PREDICT = false   -- starts the CNN in predict mode, as opposed to use it to generate representations for the LSTM
local LSTM_PREDICT = false  -- starts the LSTM in predict mode, requires CNN_PREDICT=true
local LSTM_TRAIN = true     -- runs a LSTM only training

print("vector size",GloVe:vectorSize())

require 'hyperparams'

local logger = optim.Logger('log/accuracy.log')
logger:setNames{'epoch', 'sample', 'loss'}

require 'disputils'

-- define training set
local reader_train=DataReader("train")
local dataset_train=reader_train:get_dataset()
local words=reader_train:get_words()
-- define validation set
local reader_val=DataReader("val")
local dataset_val=reader_val:get_dataset()


-----------------------------------------------------------------------------------

-- read one file as a torch image, handle b&w images
local function getImageInput(sample)

	local ok,img = pcall(image.load,sample.file)
	if not ok then
		print ("cannot load : ",sample.file)
		error()
	end	
    img = image.scale(img,SIZE,SIZE)
	local size = img:size()
    if size[1]==3 then
    	input = img:view(1,3,SIZE,SIZE)
    else
    	if size[1]==1 then
    		bw = img:view(1,1,SIZE,SIZE)
    		input = torch.cat(bw,bw,2):cat(bw,2)
    		print("processed b&w image ",sample.file)
    	else
    		print('invalid size for image)',size)
    	end
    end	
	return to_cuda(input)	 
end	

-- compute the table of word vectors for the caption target
local function computeTargetCaption(sample)
	-- do the sentence processing using GloVe, to compute expected caption prediction
    local caption_target = {}
    local last_index=0
    for i,token in pairs(sample.caption) do
	   	local word_vector = GloVe:oneHot(token) 
	   	assert (word_vector ~= nil, "nil word vector")
	   	local word_embedding = encode_layer:forward(to_cuda(word_vector))
	   	caption_target[i] = to_cuda(word_embedding):clone()
	   	last_index = i
	   	if i == MAX_SENTENCE_LENGTH then
	   		break -- truncate sentences that are too long
	   	end	
	end
	-- put an End Of Sentence vector at the end of sequence, up to max size
	-- because all the samples in a mini batch need to be of the same sequence length
	for i = last_index+1, MAX_SENTENCE_LENGTH do
		caption_target[i] = EOS
	end
	return caption_target
end	

-- predict image class according to ImageNet classes taxonomy
local function predictImageClass(sample,image_output)

    local maxs, indices = tensor.max(image_output,3)
    local predicted = indices[1]
    local word_index = predicted[1][1]
    local predicted_label = words[word_index]
    if LOG then
    	print('predicted class : '
    		  ..word_index..' : '..predicted_label..
    		  ' for path '..sample.file)
    end	

end	

-- compute the predicted caption using LSTM
local function predictImageCaption()

	-- forward prop through the LSTM
	local lstm_output
	local i = 0
    local caption_predicted = {}
	repeat
		lstm_output = lstm:forward(lstm_output:cat(image_output))
	   	caption_predicted[i] = lstm_output
		i = i + 1
	until i == MAX_SENTENCE_LENGTH 
	   or tensor.all(lstm_output:eq(EOS)) -- tensor equality 
	return caption_predicted
end	


-- this is the forward step definition funaction
function forward()
  -- forward through LSTM the image representation and target sentence
  local outputs = lstm:forward(inputs) -- 20x128x1000 -> 20:128x4000
  -- compute the loss as cross entropy
  local loss = criterion:forward(outputs, classIndexes)
  -- display target sentence and predicted sentence for a random sample of the minibatch
  display_word_outputs(outputs,labels,loss,math.random(1,BATCH_SIZE),epoch,current_lr)
  logger:add{epoch, sample_number, loss}
  return loss
end


-- this is the gradient descent step definition funaction
function forward_backward(params)
  lstm:forget()
  gradParams:zero() 
  -- forward through LSTM the image representation and target sentence
  --print  ("inputs", inputs)
  --print("inputs[1]",inputs[1]:size())
  local outputs = lstm:forward(inputs) -- 20x128x1000 -> 20:128x4000
  -- compute the loss as cross entropy
  local loss = criterion:forward(outputs, classIndexes)
  -- display target sentence and predicted sentence for a random sample of the minibatch
  display_word_outputs(outputs,labels,loss,math.random(1,BATCH_SIZE),epoch,current_lr)
  logger:add{epoch, sample_number, loss}
  -- backprop on the loss
  local gradOutputs = criterion:backward(outputs, classIndexes)
  -- backprop on the LSTM
  lstm:backward(inputs, gradOutputs) --  20x128x1000, 2x2x128x300 , return value is not used
  return loss, gradParams
end


-- process the dataset in trazining or validation mode
local function process_dataset(dataset,train)

--SAMPLE_SAVE_COUNT=BATCH_SIZE*2

	print (train and "==================== process training set" or "===================== process validation set")

	for m,sample in pairs(dataset) do

		sample_number=m

	    local ok, input =  pcall(getImageInput,sample)
	    if not ok then
			print('no image for ',sample.file,' at sample ', m)
			input = input_old
		end		
		if m == 1 then
			-- FIXME find a way to go to the next pair : here we use another image to replace the missing one
			input_old = input -- default image in case we cannot read it, otherwise the mini-batch is not complete
		end
		
		-- forward the image through the CNN to create a representation
	    local image_output = to_cuda(cnn:forward(input))    -- run the image data through the CNN
	    local target = computeTargetCaption(sample)          --  target value for LSTM training on captions
	    local caption_predicted
	 
	    if CNN_PREDICT or LSTM_PREDICT then
		    if CNN_PREDICT  then predictImageClass(sample,image_output)   end   -- predict image class as ImageNet taxonomy and print it
			if DISPLAY      then manage_display(m,input) end   -- display first 16 images for checking
			if LSTM_PREDICT then
				caption_predicted = predictImageCaption()    -- compute caption predicted for image
			end
		end	-- PREDICT

		if LSTM_TRAIN then
			local batch_index = 1 + (m - 1) % BATCH_SIZE
			local input_flat = image_output:view(CNN_OUTPUT_SIZE)
			local previous_target = input_flat:clone() -- first input is the image
			-- the preceding word is set at the LSTM input, because the LSTM should predict it
			-- in prediction mode of course the previous output of the the LSTM should be the next input
			for s = 1, math.min(#target, MAX_SENTENCE_LENGTH) do 	
				inputs[s][batch_index] = previous_target
				local label =decode_layer:forward(target[s])
				labels[s][batch_index] = label:clone()
				classIndexes[s][batch_index] = GloVe:classIndex(labels[s][batch_index]) 
				previous_target = target[s] -- here there is nother recurrence of the s-1 output to the current input
			end

			if batch_index == BATCH_SIZE then

				print("processing sample ",m)

				if train then
				   	local optimState = {learningRate = current_lr}
				   	-- do the gradient descente, optim will repetitively call the forward_backward function
				   	-- and update de parameters according to the computed gradient and the learning rate parameter
				   	local ok = pcall (optim.sgd, forward_backward, params, optimState) -- TODO : try rmsprop
				   	if not ok then
				   		print("********************* INVALID SGD FOR ",m)
					end
				   	-- save parameters in case there is a program abort
				   	if m % SAMPLE_SAVE_COUNT == 0 then
				   		--params = lstm:getParameters()
				   		torch.save('parameters/lstm-params.t7', lstm:float())
				   	end
				else
					local loss=forward()
				end

				init_tables() -- reinit mini batch tables for next mini batch
			end
		end -- LSTM_TRAIN
			
--[[if m % SAMPLE_SAVE_COUNT == 0 then
	return
end	]]--

	end -- samples

end	

-- init the LSTM networks, the coding/decoding layers, and the raining criterium
function init_lstm()

	-- LSTM model loading
	if LSTM_PREDICT or LSTM_TRAIN then

		local embedding_size = GloVe:vectorSize()
		local dict_size = GloVe:dictionarySize()

		-- create embedding layers to encode/decode word embeddings
		-- these layers are not subject to backprop
		encode_layer = nn.Linear(dict_size, embedding_size)
		decode_layer = nn.Linear(embedding_size, dict_size)
		encode_layer.weight[{ {1, embedding_size} , {1, dict_size} }] = GloVe:encoding():transpose(1,2)
		decode_layer.weight[{ {1, dict_size} , {1, embedding_size} }] = GloVe:encoding()
		encode_layer = to_cuda(encode_layer)
		decode_layer = to_cuda(decode_layer)

		-- constants
		-- BOS = to_cuda(encode_layer:forward(torch.zeros(GloVe:dictSize())))
		EOS = to_cuda(encode_layer:forward(to_cuda(GloVe:oneHot('<unk>')))):clone()

		-- create the LSTM model itself, the is an implicit assumption that image representations
		-- and word representations are of the same size
		-- so the image can be the initial word input into the LSTM
		lstm = nn.Sequencer(lstm_model.createCaptionModel(
			CNN_OUTPUT_SIZE, dict_size, embedding_size, GloVe:encoding(), MAX_SENTENCE_LENGTH, false))

		to_cuda(lstm)

		print(lstm)

		-- the loss will be the cross entropy computed after a sofmax on a sequence
		-- of one hot output word vectors
		criterion = to_cuda(nn.SequencerCriterion(nn.CrossEntropyCriterion()))

		-- set these variables to point to the LSTM parameters so as to use them in gradient descent
		params, gradParams = lstm:getParameters()
	end

	print("EOS",GloVe:token(decode_layer:forward(to_cuda(EOS:float()))))

end

-- init the data tables used for each batch
function init_tables()
	for s=1,MAX_SENTENCE_LENGTH do 
		inputs[s] = to_cuda(torch.zeros(BATCH_SIZE, CNN_OUTPUT_SIZE))
		labels[s] = to_cuda(torch.zeros(BATCH_SIZE, GloVe:dictionarySize() ))
		classIndexes[s] = to_cuda(torch.zeros(BATCH_SIZE))
	end
end


------------------- MAIN ------------------
-- cnn, encode_layer, decode_layer lstm, criterion, params, gradparams, inpit, labels, classIndexes, epoch, input_old

torch.setdefaulttensortype('torch.FloatTensor') 
torch.setnumthreads(8) 

-- resnet model loading
cnn = models.setup(CNN_PREDICT)
cnn = nn.Sequencer(cnn)

init_lstm()

-- initializations
inputs = {}
labels = {}
classIndexes = {}
sample_number=0 -- global variable sample number

init_tables()

epoch=0       -- make it global
input_old = 0 -- make it global

current_lr = LEARNING_RATE


--MAX_EPOCH=2 -- ******************

for epoch = 1,MAX_EPOCH do

	-- train on the dataset

	dataset_train = DataReader:shuffle(dataset_train) -- random shuffle samples at each epoch
	process_dataset(dataset_train,true)

	-- validate the dataset

	dataset_val = DataReader:shuffle(dataset_val) -- random shuffle samples at each epoch
	process_dataset(dataset_val,false)

	--[[if epoch % LR_DECAY_EPOCH_COUNT == 0 then
		current_lr = current_lr / 2
		print("learning rate changed to : ", current_lr)
	end	]]--
	current_lr = LEARNING_RATE / (1 + epoch / 10)
	print("learning rate changed to : ", current_lr)	

end

if LSTM_TRAIN then
	print('final save of model')
	--params = lstm:getParameters()
	local datetime=os.date("%m-%d-%y-%H-%M-%S")
	torch.save('parameters/lstm-params-final-'..datetime..'.t7', lstm:float()) -- params:float()
end

print('main ended')
