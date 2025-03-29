## 📊 Figures and Tables

### Table 1

<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th rowspan="2">Noise</th>
      <th colspan="3">Sym. 60%</th>
      <th colspan="3">Asym. 40%</th>
      <th colspan="3">Inst. 40%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Epoch</td>
      <td>20</td>
      <td>60</td>
      <td>100</td>
      <td>20</td>
      <td>60</td>
      <td>100</td>
      <td>20</td>
      <td>60</td>
      <td>100</td>
    </tr>
    <tr>
      <td>Loss</td>
      <td>0.95</td>
      <td>0.80</td>
      <td>0.63</td>
      <td>0.80</td>
      <td>0.64</td>
      <td>0.58</td>
      <td>0.84</td>
      <td>0.65</td>
      <td>0.59</td>
    </tr>
    <tr>
      <td>Wrong Event</td>
      <td>0.96</td>
      <td>0.98</td>
      <td>0.97</td>
      <td>0.94</td>
      <td>0.94</td>
      <td>0.92</td>
      <td>0.94</td>
      <td>0.95</td>
      <td>0.94</td>
    </tr>
  </tbody>
</table>

<center>Table 1. The AUC values of Loss and Wrong Event. The experiment is conducted under the setting with ResNet18, with SGD, lr = 1e-2, weight_decay = 5e-4, batch_size=128.</center>

### Figure 1

<div>			
    <center>	
    <img src="566e9264c6d2d103521b24dc031fb5c.png"
         alt=""
         style=""/>
    <br>		
    Figure 1. The AUC-ROC curves of loss and wrong event at different training stages. The experiment is conducted under the setting with ResNet18, with SGD, lr = 1e-2, weight_decay = 5e-4, batch_size=128.
    </center>
</div>




