-- autoconfiguration based on hostname
local hostname=io.popen('hostname'):read()
if string.sub(hostname,1,7) == 'yvesMBP' then
	MACHINE = 'mac'
	GPU = false
elseif hostname == 'pcybldlc' then
	MACHINE = 'pc'
	GPU = true
elseif hostname == 'yves-ml' then
	MACHINE = 'pc2'
	GPU = false
else    
    MACHINE = 'pc'
    GPU = false
end

-- gpu dependent configuration
if GPU then
	require 'torch'
    tensor = require 'cutorch'
    TENSOR_NAME = 'torch.CudaTensor'
    function to_cuda(obj) 
    	return obj:cuda()
    end	
else
    TENSOR_NAME = 'torch.FloatTensor'
	tensor = require 'torch'
    function to_cuda(obj) 
    	return obj
    end	
end