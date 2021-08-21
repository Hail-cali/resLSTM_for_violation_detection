# resLSTM_for_violation_detection
<hr />

|main | contents |
|---|:---:|
| `for` | 21 summer semester  deep learning  co-project |
| `period` | 21.08.14-21.08.20 |
| `role` |`modeling ` |
| `model` | `ResNet & LSTM `  |

>### CCTV video violation detection using *CNN-LSTM*

+ ### dataset 

![dataset preview](/img_result/img/scfd_fight.png "Aihub CCTV Fight")
  
+ ### method 

>> 

+ ### result

<img width="400" src="/img_result/A_reslstm_SFGD_acc.png" alt="result" title="res Lstm result using SCFD">,
<img width="400" src="/img_result/A_reslstm_SGFD_loss.png" alt="result" title="res Lstm result using SCFD">

>code descrition
> - **model** => `featurenet.py` 
> - **utils** => `data_loader.py` |ideo to frame torch Tensor & load dataset from dir)
> - ***test_*** => `test_A.py` | using resnet for feature extraction without backprop
> - ***test_*** => `test_B.py` | piled up layer (resnet & lstm) for backprop
> - **dataset**



