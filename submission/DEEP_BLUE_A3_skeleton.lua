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
   local output_size = torch.LongStorage(3)

   output_size[1] = input_size[1]
   output_size[2] = math.floor((input_size[2] + 1 - self.kW)/self.dW)
   output_size[3] = input_size[3]

   self.output = torch.DoubleTensor(output_size)
   
   local exp_beta_x = torch.DoubleTensor(input_size)

   exp_beta_x:mul(input, self.beta) -- multiplication and exponentiation
   exp_beta_x:exp()                 -- is only done once
   
   for batch_idx = 1,input_size[1] do
      for frame_idx = 1,input_size[3] do
         for step_idx = 1,output_size[2] do
            local window_start = (step_idx - 1)*self.dW + 1
            local window_end   = window_start + self.kW - 1
            local window_out = input[{ batch_idx, {window_start, window_end}, frame_idx }]:sum()
            window_out = window_out / self.kW
            window_out = torch.log(window_out)
            window_out = window_out / self.beta
           
            self.output[{ batch_idx, step_idx, frame_idx }] = window_out
         end -- end: feature vector loop
      end -- end: frame loop
   end -- end: minibatch loop
   --------- END: OUR CODE

   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   
   --------- OUR CODE
   local in_size = input:size()
   local out_size = gradOutput:size()

   self.gradInput = torch.zeros(in_size)
   local exp_beta_x = input:clone():mul(self.beta):exp() -- only compute once

   for batch_idx=1,out_size[1] do
      for frame_idx=1,out_size[3] do
         for step_idx=1,out_size[2] do
            local gradInput_win_start = (step_idx - 1)*self.dW + 1
            local gradInput_win_end   = gradInput_win_start + self.kW - 1
            local denom_sum = exp_beta_x[{ batch_idx, {gradInput_win_start, gradInput_win_end}, frame_idx }]:sum()

            local dOut_dIn = exp_beta_x[{batch_idx, {gradInput_win_start, gradInput_win_end}, frame_idx}]:clone():div(denom_sum)
            self.gradInput[{ batch_idx, {gradInput_win_start, gradInput_win_end}, frame_idx }]:add(dOut_dIn:mul(gradOutput[{batch_idx, step_idx, frame_idx}]))
         end
      end
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
