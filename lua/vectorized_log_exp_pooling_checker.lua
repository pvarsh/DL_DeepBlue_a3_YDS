-------------------------------------------------
-- Runs a sample input forward and backward
-- through nn.TemporalLogExpPooling
-------------------------------------------------

require 'nn'
require 'temporalLogExpPooling'

torch.manualSeed(123)
x = torch.rand(4,8,2)
model = nn.Sequential()
model:add(nn.TemporalLogExpPooling(3,1,10))
print(model:forward(x))

gradOut = torch.rand(4,6,2)
print(model:backward(x, gradOut))