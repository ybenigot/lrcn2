require 'torch'
require 'nn'
require 'rnn'
x1=torch.ones(4)
x1[2]=2
x1[3]=3
x1[4]=4
X=torch.Tensor(5,4)
X[1]=x1
X[2]=2*x1
X[3]=3*x1
X[4]=4*x1
X[5]=5*x1
Y=torch.Tensor(5,4)
Y:copy(X)
Y[1]=1
Y[2]=-1
Y[3]=1
Y[3][2]=0
Y[3][4]=0
Y[4][1]=0
Y[4][3]=0
Y[5][1]=0
Y[5][2]=0
print(X,Y)
print('size',X:size(),Y:size())

XX={X,X,X,X,X,X} -- a table of 6
YY={Y,Y,Y,Y,Y,Y} -- a table of 6

model = nn.Sequential()
model:add(nn.LSTM(4, 3, 16))
model = nn.Sequencer(model)

model:zeroGradParameters() 

outputs = model:forward(XX)

c1=nn.CosineEmbeddingCriterion()
C=nn.SequencerCriterion(c1)

loss=C:forward({outputs, YY}, {torch.ones(5),torch.ones(5)})
print("loss: ",loss)
print("counts : ",#XX,#outputs,#YY)

dloss_doutputs = C:backward({outputs,YY}, {torch.ones(5),torch.ones(5)})
--dloss_doutputs = C:backward(outputs, YY)
print("backward result : ",dloss_doutputs[1][1],dloss_doutputs[2][1]) 

-- select entry 1 of size batchsize x output then entry 1 for outputs
-- apply gradient to all the sequence elements ?
gradInput = model:backward(XX, {dloss_doutputs[1][1],dloss_doutputs[1][1],dloss_doutputs[1][1],dloss_doutputs[1][1],dloss_doutputs[1][1],dloss_doutputs[1][1]}) 

