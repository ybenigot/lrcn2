-- data reader for coco, multithread, with caption reading

require 'io'
require 'hwconfig'

local M = {}
local DataReader = torch.class('DataReader', M)

local WORD_FILE='webapp/words1000.txt'

local dataset 
local words -- the dictionnary of labels corresponding to an output value of the cnn

function DataReader:load_images(dataset_type)

    local n=1;
    local DATASET_INFO = 'dataset/coco-caption-'..dataset_type..'-'..MACHINE..'.txt'

	for line in io.lines(DATASET_INFO) do
		local first = true
	    local filepath=nil
    	local sentence={}
		for token in string.gmatch(line, "[^%s]+") do
			if first then
				filepath = token
				first=false
			else
				table.insert(sentence,token)
			end	
		end
		--print('filepath : ',filepath)
		--print('caption : ' ,sentence) 
		self.dataset[n]={}
		self.dataset[n]['file'] = filepath
		self.dataset[n]['caption'] = sentence
		n = n + 1
	end	
end

-- load index to prediction label table from,imagenet word file
function DataReader:load_words()
	local i=1
	for line in io.lines(WORD_FILE) do
	  self.words[i] = line
	  i = i + 1
	end  
	print('read '..(i-1)..' lines')
end

function DataReader:__init(dataset_type)
	self.dataset = {}
	self.words = {}
	self:load_images(dataset_type)
	self:load_words() -- imagenet correspondence table for transfert learning
end


function DataReader:get_dataset() 
	return self.dataset
end

-- see: http://en.wikipedia.org/wiki/Fisher-Yates_shuffle

function DataReader:shuffle(t)
  local n = #t
 
  while n >= 2 do  -- n is the last pertinent index
    local k = math.random(n) -- 1 <= k <= n
    t[n], t[k] = t[k], t[n]  -- switch n and k
    n = n - 1
  end
  return t
end

-- returns the word table for imagenet, not MScoco, useful only for transfert learning use case
function DataReader:get_words() 
	return self.words
end

return M.DataReader
