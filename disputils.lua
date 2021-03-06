
require 'paths'
require 'image'
require 'io'
require 'table'

local GloVe = require 'glove/glove'

local M={}

if GPU then
    tensor = require 'cutorch'
    TENSOR_NAME = 'torch.CudaTensor'
else
	tensor = require 'torch'
end

-- prepare image table before displaying it
local function accumulate_images(input)

    local a=1+math.floor((m-1)/4)
    local b=1+(m-1)%4
    --print(a,b)
    img2=tensor.Tensor(input:size()):copy(input)
    img2 = tensor.reshape(img2,3,SIZE,SIZE)
    img2 = image.drawText(img2, predicted_label, 10, 10, {color = {255, 255, 0}, size = 2})
    image_tensor[a][b] = img2
end	

-- show images as a 4x4 composite image
local function display_images(image_tensor)

	local image_composite
	for row=1,4 do
		local img_row = tensor.cat(image_tensor[row][1],image_tensor[row][2],1):cat(image_tensor[row][3],1):cat(image_tensor[row][4],1)
		--print(img_row:size())
		if row == 1 then
			image_composite = img_row
		else
			image_composite = image_composite:cat(img_row,1)
		end		
	end

	-- create and display image mosaic
	image_composite = tensor.reshape(image_composite,4,4,3,SIZE,SIZE)
	image_composite = image_composite:transpose(1,3)
	image_composite = image_composite:transpose(3,4)
	image_composite = tensor.reshape(image_composite,3,SIZE*4,SIZE*4)
	---print(image_composite:size())
	tensor.reshape(image_composite,3,SIZE*4,SIZE*4)
	image.display(image_composite)

end	

-- prepare the display of a set of 16 images for test purposes
function manage_display(m,input)
    if (m <= 16) then
		accumulate_images(input)
    end	
	if m == 16 then
		display_images(image_tensor)
	end	
end	

function vector2csv(vec)
  return table.concat(vec:totable(),',')
end 

local file = io.open("log/trace-disputils.txt", 'w') -- file we be closed implicitly at the end of the program
local old_epoch=0

-- display first batch item captions computed/expected
function display_word_outputs(outputs,targets,loss,batch_index,epoch,lr)
	local target_sentence
	local computed_sentence
	if epoch ~= old_epoch then
		old_epoch=epoch
		file:write(" epoch: "..epoch..", learning rate: "..lr.."\n")
	end	
	for i=1,MAX_SENTENCE_LENGTH do
		local delta = outputs[i] - targets[i]
		if i == 1 then
			target_sentence   = GloVe:token(targets[i][1]:float())
			computed_sentence = GloVe:token(outputs[i][1]:float())
		else
			target_sentence   =   target_sentence..' '..GloVe:token(targets[i][1]:float())
			computed_sentence = computed_sentence..' '..GloVe:token(outputs[i][1]:float())
		end
		local current_vector = outputs[i][1]:float()
		previous_vector = current_vector
	end	

	print ("    target sentence : ",target_sentence)	
	print ("computed",batch_index," : ",computed_sentence)
	print ("batch loss : ",loss)
	print ("-------------------------------------")

	file:write("    target sentence : "..target_sentence.."\n")
	file:write("computed"..batch_index.." : "..computed_sentence.."\n")
	file:write ("batch loss : "..loss.."\n")
	
end

function print_tensor(name,t)
	print ("tensor", name, t:type(), t:size())
	u = t:view(t:nElement())
	for i = 1,math.min(10,u:nElement()) do
		print(u[i])
	end
end	

return M
