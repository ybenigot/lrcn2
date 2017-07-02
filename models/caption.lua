-- language model for captioning
--
-- in :
--   imagerepresentation
--
-- out :
--   predicted sentence for image

--require 'torch'
require 'nn'
require 'rnn'
require 'nngraph'


--module("caption", package.seeall)
local M={}

-- freeze a layer by overloading its bacprop methods
local function freezeLayer(layer)
	layer.accGradParameters = function() end
	layer.updateParameters = function() end
end

-- GloVec:distance(v,1)[2] provides backward word vector to word correspondance

-- image_size1 : size of one hot word vector input = size of image representation
-- dict_size : size of the dictionnary
-- embedding_size : size of embedding vectors
-- vectors : glove vectors
-- rho : number of steps for state propagation
-- prediction : true if this is not training
function M.createCaptionModel(image_size, dict_size, embedding_size, vectors, rho, prediction, lstm1_file, lstm2_file, format)

 	print("createmodel : ",image_size,dict_size,embedding_size,vectors:size(),rho,prediction,lstm1_file,lstm2_file,format)

 	local model = nn.Sequential()

	nn.FastLSTM.usenngraph = false
	nn.FastLSTM.bn = false -- normalize hidden values

    assert (image_size == embedding_size,"assumption image_size = embedding_size not satisfied")

	-- create model : two layer LSTM
	if lstm1_file == nil then
		model:add(nn.FastLSTM(embedding_size, embedding_size, rho))
	else
		print("read file",lstm1_file,"in format",format)
		local lstm1 = torch.load(lstm1_file,format)
		model:add(lstm1)
	end		
	if lstm2_file == nil then
		model:add(nn.FastLSTM(embedding_size, embedding_size, rho))
	else
		print("read file",lstm2_file,"in format",format)
		local lstm2 = torch.load(lstm2_file,format)
		model:add(lstm2)
	end
	
	local layer4 = nn.Linear(embedding_size, dict_size)
	layer4.weight = vectors
	-- fixe the weights of the linear layers
	freezeLayer(layer4)

	model:add(layer4) -- a decoding layer to convert embeddings to one hot, not learned

	if prediction then
		model:add(nn.LogSoftMax()) 
	end	

	to_cuda(model)

	return model

end

return M