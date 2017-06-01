-- adapted for prediction on LRCN (from Facebook code on resnets)
-- references :
--    Facebook resnet code : https://github.com/facebook/fb.resnet.tensor
--    LRCN : http://jeffdonahue.com/lrcn/
--           https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&cad=rja&uact=8&ved=0ahUKEwiOhd64hLvTAhUB0hoKHfxvAhEQFgg6MAM&url=https%3A%2F%2Farxiv.org%2Fabs%2F1411.4389&usg=AFQjCNFgyGMHQYYPP-QSp4Y0FA0q2gdvDg
--
-- this code can do :
--    LRCN training, with a freezed resnet        use LSTM_TRAIN  = true
--    resnet prediction (for testing purposes)    use CNN_PREDICT = TRUE
--    LRCN prediction                             use LSTM_PREDICT = true

require 'paths'
require 'nn'
require 'rnn'
require 'image'
require 'optim'
require 'io'

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

-- constants
local BOS = to_cuda(torch.zeros(GloVe.vector_size()))
local EOS = GloVe:word2vec(".",false)

print("vector size",GloVe.vector_size())

require 'hyperparams'

local logger = optim.Logger('accuracy.log')
logger:setNames{'epoch', 'sample', 'loss'}

require 'disputils'

-- data loading
local reader=DataReader.create()
local dataset=reader.get_dataset()
local words=reader.get_words()

-----------------------------------------------------------------------------------

-- read one file as a torch image, handle b&w images
local function get_image_input(sample)

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
    	else
    		print('invalid size for image)',size)
    	end
    end	
	return to_cuda(input)	 
end	

-- compute the table of word vectors for the caption target
local function compute_target_caption(sample)
	-- do the sentence processing using GloVe, to compute expected caption prediction
    local caption_target = {}
    local last_index=0
    -- print(sample.caption)
    for i,token in pairs(sample.caption) do
	   	local word_vector = GloVe:word2vec(token,false) 
	   	assert (word_vector ~= nil, "nil word vector")
	   	caption_target[i] = word_vector
	   	last_index = i
		--print ("      initial word : ",token)	
		--print ("reconstructed word : ",GloVe:vec2word(word_vector))
		--print ("--------------------------------------")
	end
	-- put a zero vector = EOS at the end of sequence, up to max size
	--[[for i = last_index+1, MAX_SENTENCE_LENGTH do
		caption_target[i] = EOS 
	end]]--
	caption_target[last_index+1] = EOS -- add a point at the end of sentence
	return caption_target
end	

-- predict image class according to ImageNet classes taxonomy
local function predict_image_class(sample,image_output)

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
local function predict_image_caption()

	-- forward prop through the LSTM
	local lstm_output = BOS
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

------------------- MAIN ------------------

torch.setdefaulttensortype('torch.FloatTensor') 
torch.setnumthreads(8) 

-- resnet model loading
local cnn,cnn_output_size = models.setup(CNN_PREDICT)
cnn=nn.Sequencer(cnn)
local LSTM_INPUT_SIZE=cnn_output_size + GloVe:vector_size()

-- optim parameters
local params, gradParams 

-- LSTM model loading
if LSTM_PREDICT or LSTM_TRAIN then
  lstm = nn.Sequencer(lstm_model.createCaptionModel(
  	LSTM_INPUT_SIZE, GloVe:vector_size(),MAX_SENTENCE_LENGTH))
  print(lstm)
  local base_criterion = nn.MSECriterion()
  base_criterion.sizeAverage=false
  criterion = nn.SequencerCriterion(base_criterion)
  to_cuda(criterion)
  params, gradParams = lstm:getParameters()
end

print("BOS",GloVe:vec2word(BOS:float()))


-- initializations
local image_tensor = {}
for i=1,4 do image_tensor[i]={} end
local BatchInputTable = {}
local BatchLabelTable = {}
local function init_tables()
	for s=1,MAX_SENTENCE_LENGTH do 
		BatchInputTable[s] = to_cuda(torch.zeros(BATCH_SIZE, LSTM_INPUT_SIZE))
		BatchLabelTable[s] = to_cuda(torch.zeros(BATCH_SIZE, GloVe:vector_size()))
	end
end
init_tables()

-- process dataset
for epoch = 1,MAX_EPOCH do
	dataset = DataReader:shuffle(dataset) -- random shuffle samples at each epoch
	for m,sample in pairs(dataset) do

		--[[if m > (1 * BATCH_SIZE) then
			break -- testing on a little dataset
		end]]--	

	    ok, input =  pcall(get_image_input,sample)
	    if not ok then
			print('no image for ',sample.file,' at sample ', m)
		else		
		    local image_output_raw = cnn:forward(input):float()    -- run the image data through the CNN
			local image_output = to_cuda(image_output_raw)     		    
		    local target = compute_target_caption(sample)          --  target value for LSTM training on captions
		 
		    if CNN_PREDICT or LSTM_PREDICT then
			    if CNN_PREDICT  then predict_image_class(sample,image_output)   end   -- predict image class as ImageNet taxonomy and print it
				if DISPLAY      then manage_display(m,input) end   -- display first 16 images for checking
				if LSTM_PREDICT then
					caption_predicted = predict_image_caption()    -- compute caption predicted for image
				end
			end	-- PREDICT

			if LSTM_TRAIN then
				local batch_index = 1 + (m - 1) % BATCH_SIZE
				--print("batch_index",batch_index)

				local previous_target = BOS
				local input_flat = to_cuda(image_output:view(CNN_OUTPUT_SIZE))
				for s = 1, math.min(#target, MAX_SENTENCE_LENGTH) do						--print_tensor('image_output',image_output)
					BatchInputTable[s][batch_index] = input_flat:cat(to_cuda(previous_target))
					BatchLabelTable[s][batch_index] = target[s]	
					previous_target = target[s]				
					--print_tensor("previous_target",previous_target)
					--print_tensor('batch input',BatchInputTable[s][batch_index])				
				end

				if batch_index == BATCH_SIZE then
				   	function forward_backward(params)
				      gradParams:zero() 
				      local outputs = lstm:forward(BatchInputTable) -- 20x128x1000 -> 20:128x300
				      local loss = criterion:forward(outputs, BatchLabelTable)
				      display_word_outputs(outputs,BatchLabelTable,loss,math.random(1,BATCH_SIZE))
				      logger:add{epoch, m, loss}
				      local gradOutputs = criterion:backward(outputs, BatchLabelTable)
				      local gradInputs = lstm:backward(BatchInputTable, gradOutputs) --  20x128x1000, 2x2x128x300
				      return loss, gradParams
				   	end
				   	local optimState = {learningRate = LEARNING_RATE}
				   	print_tensor("grad",gradParams)
				   	print_tensor("params",params)
				   	optim.sgd(forward_backward, params, optimState) -- TODO : try rmsprop
				   	if m % SAMPLE_SAVE_COUNT == 0 then
				   		params = lstm:getParameters()
				   		torch.save('lstm-params.t7', params:float())
				   	end
					init_tables()
				end
			end -- LSTM_TRAIN
		end	
	end -- samples
	if epoch % LR_DECAY_EPOCH_COUNT == 0 then
		LEARNING_RATE = LEARNING_RATE / 2
		print("learning rate changed to : ", LEARNING_RATE)
	end	
end

if LSTM_TRAIN then
	print('final save of model')
	params = lstm:getParameters()
	torch.save('lstm-params-final.t7', params:float())
end

print('main ended')

--[[
--  criterion = nn.SequencerCriterion(nn.CosineEmbeddingCriterion())
				    --local YVALUES = {to_cuda(torch.ones(BATCH_SIZE)),to_cuda(torch.ones(BATCH_SIZE))}
				      -- local loss = criterion:forward{outputs, BatchLabelTable}, YVALUES)
				      --local dloss_doutputs = criterion:backward{outputs, BatchLabelTable}, YVALUES)
				      --local dloss = {}
				      -- we use the same gradient for all time steps, because the parameters are common for all time step
				      --for i=1,#BatchInputTable do
				      --    dloss[i] = dloss_doutputs[1][1] -- FIXME 1 1
				      --end    


]]--



