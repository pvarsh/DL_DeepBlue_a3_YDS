-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Please find submission instructions on the handout
------------------------------------------------------------------------

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)

   --------- OUR CODE
   local input_size = input:size()
   local output_size = torch.LongStorage(2)
   output_size[1] = math.floor((input_size[1] - self.kW)/self.dW + 1)
   output_size[2] = input_size[2]

   self.output = torch.DoubleTensor(output_size)
   -- print('self.output:size()')
   -- print(self.output)
   local exp_beta_x = torch.DoubleTensor(input_size)

   exp_beta_x:mul(input, self.beta) -- multiplication and exponentiation
   exp_beta_x:exp()                 -- only is done once
   -- print('updateOutput:: exp_beta_x')
   -- print(exp_beta_x)
   
   local pos = 1
   local count = 1
   while pos + (self.kW - 1) <= input_size[1] do
      self.output[count] = exp_beta_x[{ {pos,(pos + self.kW - 1)} }]:sum(1)
      pos = pos + self.dW
      count = count + 1
   end
   ((self.output:mul(1/self.kW)):log()):mul(1/self.beta)
   --------- END: OUR CODE

   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   
   --------- OUR CODE
   local input_size = input:size()
   
   -- Precompute exp(beta * input)
   local exp_beta_input = torch.exp(torch.mul(input, self.beta))

   -- Declare tensor for sliding window derivative
   local denom_sum = torch.Tensor(1, input_size[2])
   local dOut_dIn_window = torch.Tensor(self.kW, input_size[2])

   -- Reset self.gradInput to zeros
   -- TODO: check if self.gradInput is already of correct dimension and
   --       reset to zero. This might be better memory management.
   self.gradInput = torch.zeros(input_size)

   -- Set loop indexes
   local pos = 1   -- index of first row/element in pooling window
   local count = 1 -- index of row/element in gradOutput

   while pos + self.kW - 1 <= input_size[1] do

      -- Compute sliding window derivative
      dOut_dIn_window[{}] = exp_beta_input[{ {pos, pos + self.kW - 1} }]
      denom_sum[{}] = dOut_dIn_window:sum(1)

      -- Loop through columns of dOut_dIn_window
      for col_idx=1,input_size[2] do
         -- Divide exp(beta input) by denominator sum
         -- After this dOut_dIn_window contains dOut_dIn derivative
         dOut_dIn_window[{ {},{col_idx} }]:div(denom_sum[1][col_idx])         

         -- Add dE/dx_{i} * dx_{i}/dx_{i-1} to self.gradInput
         -- TODO: rewrite using several lines for readability
         self.gradInput[{ {pos,pos+self.kW-1},{col_idx} }]:add(dOut_dIn_window[{ {},{col_idx} }]:mul(gradOutput[count][col_idx]))
      end

      -- Update loop counters
      count = count + 1
      pos = pos + self.dW
   end
   --------- END: OUR CODE
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
