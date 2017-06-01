-- language model for captioning
--
-- in :
--   imagerepresentation
--
-- out :
--   predicted sentence for image

--require 'torch'
require 'rnn'
require 'nngraph'


--module("caption", package.seeall)
local M={}


-- GloVec:distance(v,1)[2] provides backward word vector to word correspondance

function M.createCaptionModel(input_size,output_size,rho)

 	print("createmodel : ",input_size,output_size)

 	local model = nn.Sequential()

	nn.FastLSTM.usenngraph = true -- faster
	nn.FastLSTM.bn = true -- normalize hidden values

  model:add(nn.FastLSTM(input_size,  output_size, rho))
  model:add(nn.FastLSTM(output_size, output_size, rho))

  to_cuda(model)

  return model

end

return M