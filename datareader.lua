-- data reader for coco, multithread, with caption reading

require 'io'

local M = {}
local DataReader = torch.class('lrcn.DataReader', M)

if MACHINE == 'pc' then
    DATASET_INFO = 'coco-caption-pc.txt'
elseif MACHINE == 'mac' then
    DATASET_INFO = 'coco-caption-mac.txt'
elseif MACHINE == 'pc2' then
    DATASET_INFO = 'coco-caption-pc2.txt'
else
    DATASET_INFO = 'coco-caption-pc.txt'
end    

local WORD_FILE='webapp/words1000.txt'


function DataReader.create()
   return M.DataReader()
end

local dataset = {}
local words={} -- the dictionnary of labels corresponding to an output value of the cnn


local function load_images()

    local n=1;

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
		dataset[n]={}
		dataset[n]['file'] = filepath
		dataset[n]['caption'] = sentence
		n = n + 1
	end	
end

-- load index to prediction label table from,imagenet word file
local function load_words()
	local i=1
	for line in io.lines(WORD_FILE) do
	  words[i] = line
	  i = i + 1
	end  
	print('read '..(i-1)..' lines')
end

function DataReader:__init()
	load_images()
	load_words() -- imagenet correspondence table for transfert learning
end


function DataReader:get_dataset() 
	return dataset
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
	return words
end

return M.DataReader
