### Linear model (linear_baseline, tfidf)
- date: 2015/04/10
- **testing on testing data**
- commit: e9cc8a360ff75ce302166cc48780e30846c69ee8
- call: th A3_main.lua -glovePath "../glove/" -dataPath "../data/train.t7b" -nTrainDocs 100000 -nTestDocs 10000 -learningRate 0.2 -nEpochs 100 -inputDim 300
- Num training docs: 100,000 * 5
- Num testing docs: 10,000 * 5
- Word vector dimension: 300

###### Model
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
  (1): nn.Reshape(300)
  (2): nn.Linear(300 -> 600)
  (3): nn.ReLU
  (4): nn.Dropout
  (5): nn.Linear(600 -> 5)
  (6): nn.LogSoftMax
}

###### Training 
Training...	
epoch 	1	 error: 	0.79378	
epoch 	2	 error: 	0.73464	
epoch 	3	 error: 	0.76192	
epoch 	4	 error: 	0.70152	
epoch 	5	 error: 	0.68434	
epoch 	6	 error: 	0.60878	
epoch 	7	 error: 	0.63374	
epoch 	8	 error: 	0.62394	
epoch 	9	 error: 	0.63892	
epoch 	10	 error: 	0.6397	
epoch 	11	 error: 	0.57792	
epoch 	12	 error: 	0.58272	
epoch 	13	 error: 	0.62574	
epoch 	14	 error: 	0.57116	
epoch 	15	 error: 	0.60684	
epoch 	16	 error: 	0.57508	
epoch 	17	 error: 	0.5886	
epoch 	18	 error: 	0.5622	
epoch 	19	 error: 	0.56616	
epoch 	20	 error: 	0.58156	
epoch 	21	 error: 	0.58864	
epoch 	22	 error: 	0.57092	
epoch 	23	 error: 	0.56124	
epoch 	24	 error: 	0.56604	
epoch 	25	 error: 	0.5791	
epoch 	26	 error: 	0.58908	
epoch 	27	 error: 	0.57204	
epoch 	28	 error: 	0.56852	
epoch 	29	 error: 	0.56706	
epoch 	30	 error: 	0.56734	
epoch 	31	 error: 	0.56238	
epoch 	32	 error: 	0.56434	
epoch 	33	 error: 	0.56592	
epoch 	34	 error: 	0.5678	
epoch 	35	 error: 	0.5677	
epoch 	36	 error: 	0.57474	
epoch 	37	 error: 	0.58062	
epoch 	38	 error: 	0.5868	
epoch 	39	 error: 	0.56126	
epoch 	40	 error: 	0.573	
epoch 	41	 error: 	0.56158	
epoch 	42	 error: 	0.56482	
epoch 	43	 error: 	0.56414	
epoch 	44	 error: 	0.56736	
epoch 	45	 error: 	0.56962	
epoch 	46	 error: 	0.5662	
epoch 	47	 error: 	0.58388	
epoch 	48	 error: 	0.58596	
epoch 	49	 error: 	0.56824	
epoch 	50	 error: 	0.5742	
epoch 	51	 error: 	0.57442	
epoch 	52	 error: 	0.58056	
epoch 	53	 error: 	0.56176	
epoch 	54	 error: 	0.56402	
epoch 	55	 error: 	0.56846	
epoch 	56	 error: 	0.56794	
epoch 	57	 error: 	0.5707	
epoch 	58	 error: 	0.56166	
epoch 	59	 error: 	0.57048	
epoch 	60	 error: 	0.57102	
epoch 	61	 error: 	0.57546	
epoch 	62	 error: 	0.56912	
epoch 	63	 error: 	0.56654	
epoch 	64	 error: 	0.56728	
epoch 	65	 error: 	0.56912	
epoch 	66	 error: 	0.56638	
epoch 	67	 error: 	0.56922	
epoch 	68	 error: 	0.57074	
epoch 	69	 error: 	0.57816	
epoch 	70	 error: 	0.56042	
epoch 	71	 error: 	0.56212	
epoch 	72	 error: 	0.566	
epoch 	73	 error: 	0.5622	
epoch 	74	 error: 	0.5674	
epoch 	75	 error: 	0.56474	
epoch 	76	 error: 	0.57488	
epoch 	77	 error: 	0.57038	
epoch 	78	 error: 	0.56724	
epoch 	79	 error: 	0.5632	
epoch 	80	 error: 	0.56766	
epoch 	81	 error: 	0.58342	
epoch 	82	 error: 	0.57354	
epoch 	83	 error: 	0.56368	
epoch 	84	 error: 	0.57146	
epoch 	85	 error: 	0.57506	
epoch 	86	 error: 	0.56842	
epoch 	87	 error: 	0.56352	
epoch 	88	 error: 	0.56976	
epoch 	89	 error: 	0.56168	
epoch 	90	 error: 	0.56828	
epoch 	91	 error: 	0.5707	
/Users/petervarshavsky/torch/install/bin/luajit: not enough memory


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
