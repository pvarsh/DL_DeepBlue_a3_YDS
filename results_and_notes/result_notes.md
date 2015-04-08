### Linear model (linear_baseline)
- date: 2015/04/08
- **testing on testing data**
- commit: a94d973d44cf9378c9bbb6bae41b999d5b68aebf
- call: th A3_main.lua -model linear_baseline -nEpochs 30 -nTrainDocs 100000 -nTestDocs 10000 -minibatchSize 1 -inputDim 300
- Num training docs: 100,000 * 5
- Num test docs: 10,000 * 5
- Word vector dimension: 300

###### Training:
Training... 
epoch   1    error:     0.5071  
epoch   2    error:     0.49906 
epoch   3    error:     0.49764 

### Linear model (linear_baseline)
- date: 2015/04/08
- **testing on testing data**
- commit: a94d973d44cf9378c9bbb6bae41b999d5b68aebf
- call: th A3_main.lua -model linear_baseline -nEpochs 30 -nTestDocs 10000 -nTrainDocs 50000 -minibatchSize 1
- Num training docs: 50000 * nClasses
- Num test docs: 10000 * nClasses
- Word vector dimension: 50

###### Training...
epoch   1    error:     0.58884 
epoch   2    error:     0.5834  
epoch   3    error:     0.5753  
epoch   4    error:     0.57352 
epoch   5    error:     0.5729  
epoch   6    error:     0.5723  
epoch   7    error:     0.57496 
epoch   8    error:     0.57084 
epoch   9    error:     0.57064 
epoch   10   error:     0.56956 
epoch   11   error:     0.57078 
epoch   12   error:     0.56976 
epoch   13   error:     0.57096 
epoch   14   error:     0.57046 
epoch   15   error:     0.56906 
epoch   16   error:     0.57004 
epoch   17   error:     0.56924 
epoch   18   error:     0.56874 
epoch   19   error:     0.56996 
epoch   20   error:     0.5682  
epoch   21   error:     0.5683  
epoch   22   error:     0.56816 
epoch   23   error:     0.56924 
epoch   24   error:     0.56862 
epoch   25   error:     0.56736 
epoch   26   error:     0.5676  
epoch   27   error:     0.56864 
epoch   28   error:     0.56766 
epoch   29   error:     0.56722 
epoch   30   error:     0.56686 


### Linear model with ReLU
- date: 2015/04/06
- **testing on training data**
- commit: b716d8f76d9a89352b6f7fe69b187a8094511c50
- call: th A3_baseline.lua -nEpochs 50 -minibatchSize 1 -learningRate 0.2
- results: gets stuck near 56.6% error

###### Training...	
- epoch 	1	 error: 	0.60368	
- epoch 	2	 error: 	0.58508	
- epoch 	3	 error: 	0.58296	
- epoch 	4	 error: 	0.58168	
- epoch 	5	 error: 	0.57304	
- epoch 	6	 error: 	0.57966	
- epoch 	7	 error: 	0.56832	
- epoch 	8	 error: 	0.57102	
- epoch 	9	 error: 	0.56776	
- epoch 	10	 error: 	0.56448	
- epoch 	11	 error: 	0.56894	
- epoch 	12	 error: 	0.56394	
- epoch 	13	 error: 	0.56814	
- epoch 	14	 error: 	0.56666	
- epoch 	15	 error: 	0.57282	
- epoch 	16	 error: 	0.56632	
- epoch 	17	 error: 	0.56502	
- epoch 	18	 error: 	0.56498	
- epoch 	19	 error: 	0.56812

### Linear model with ReLU
- date: 2015/04/06
- **testing on training data**
- commit: b716d8f76d9a89352b6f7fe69b187a8094511c50
- call: th A3_baseline.lua -nEpochs 50 -minibatchSize 1 -learningRate 0.1
- results: gets stuck near 56.6% error

###### Training...	
- epoch 	1	 error: 	0.59566	
- epoch 	2	 error: 	0.5922	
- epoch 	3	 error: 	0.57618	
- epoch 	4	 error: 	0.57638	
- epoch 	5	 error: 	0.57248	
- epoch 	6	 error: 	0.56874	
- epoch 	7	 error: 	0.57236	
- epoch 	8	 error: 	0.56898	
- epoch 	9	 error: 	0.56924	
- epoch 	10	 error: 	0.56912	
- epoch 	11	 error: 	0.56784	
- epoch 	12	 error: 	0.56688	
- epoch 	13	 error: 	0.56602	
- epoch 	14	 error: 	0.56632	
- epoch 	15	 error: 	0.5684	
- epoch 	16	 error: 	0.5662	
- epoch 	17	 error: 	0.56562	
- epoch 	18	 error: 	0.56598	
- epoch 	19	 error: 	0.56444	
- epoch 	20	 error: 	0.56692	
- epoch 	21	 error: 	0.5648	
- epoch 	22	 error: 	0.56632	
- epoch 	23	 error: 	0.56532	
- epoch 	24	 error: 	0.5637	
- epoch 	25	 error: 	0.56482	
- epoch 	26	 error: 	0.56398	
- epoch 	27	 error: 	0.56378	
- epoch 	28	 error: 	0.56358	
- epoch 	29	 error: 	0.56382	
- epoch 	30	 error: 	0.56332	
- epoch 	31	 error: 	0.5644	
- epoch 	32	 error: 	0.56366	
- epoch 	33	 error: 	0.56404

### Linear model with ReLU and dropout
- date: 2015/04/06
- **testing on training data**
- commit: 8be9e28ca09fc67c99e7462f5311eb62681946f5
- call: th A3_baseline.lua -nEpochs 50 -minibatchSize 1 -learningRate 0.1

###### Training...	
- epoch 	1	 error: 	0.63612	
- epoch 	2	 error: 	0.59334	
- epoch 	3	 error: 	0.58454	
- epoch 	4	 error: 	0.58484	
- epoch 	5	 error: 	0.57626	
- epoch 	6	 error: 	0.57654	
- epoch 	7	 error: 	0.58172	
- epoch 	8	 error: 	0.57236	
- epoch 	9	 error: 	0.57188	
- epoch 	10	 error: 	0.57264	
- epoch 	11	 error: 	0.5724	
- epoch 	12	 error: 	0.57464	
- epoch 	13	 error: 	0.57496	
- epoch 	14	 error: 	0.5705	
- epoch 	15	 error: 	0.57372	
- epoch 	16	 error: 	0.57498	
- epoch 	17	 error: 	0.57286	
- epoch 	18	 error: 	0.57386	
- epoch 	19	 error: 	0.57172	
- epoch 	20	 error: 	0.57286	
- epoch 	21	 error: 	0.5702

### Linear model with ReLU, dropout, hidden_size = input_size * 2
- Note: this hidden layer is smaller than previous runs above
- date: 2015/04/06
- **testing on training data**
- commit: f69b290c8f5b56195f7f3dc7079fa180e657bcd8
- call: th A3_baseline.lua -nEpochs 50 -minibatchSize 1 -learningRate 0.1

###### Training...	
- epoch 	1	 error: 	0.63592	
- epoch 	2	 error: 	0.60608	
- epoch 	3	 error: 	0.60474	
- epoch 	4	 error: 	0.5896	
- epoch 	5	 error: 	0.58894	
- epoch 	6	 error: 	0.58482	
- epoch 	7	 error: 	0.58698	
- epoch 	8	 error: 	0.5866	
- epoch 	9	 error: 	0.58306	
- epoch 	10	 error: 	0.58326	
- epoch 	11	 error: 	0.58664	
- epoch 	12	 error: 	0.5878	
- epoch 	13	 error: 	0.58052	
- epoch 	14	 error: 	0.58084	
- epoch 	15	 error: 	0.5798	
- epoch 	16	 error: 	0.5813	
- epoch 	17	 error: 	0.58254	
- epoch 	18	 error: 	0.58038	
- epoch 	19	 error: 	0.57742	
- epoch 	20	 error: 	0.57848	
- epoch 	21	 error: 	0.57828	
- epoch 	22	 error: 	0.57766	
- epoch 	23	 error: 	0.57678	
- epoch 	24	 error: 	0.57882	
- epoch 	25	 error: 	0.57952	
- epoch 	26	 error: 	0.58256	
- epoch 	27	 error: 	0.5771	
- epoch 	28	 error: 	0.57968	
- epoch 	29	 error: 	0.5803	
- epoch 	30	 error: 	0.57958	
- epoch 	31	 error: 	0.5791	
- epoch 	32	 error: 	0.5764	
- epoch 	33	 error: 	0.57498
