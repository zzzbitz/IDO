## 📊 Figures and Tables

### Table 1

<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th rowspan="2">Noise</th>
      <th colspan="3">Sym. 60%</th>
      <th colspan="3">Asym. 40%</th>
      <th colspan="3">Inst. 40%</th>
      <th rowspan="3">Description</th>
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
      <td rowspan="3">
        The AUC values of Loss and Wrong Event. The experiment is conducted with ResNet18 on CIFAR10 under Sym. 60%, Asym. 40%, Inst. 40%. <br> 
        It is conducted with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=128.
      </td>
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

### Table 2

<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th rowspan="2">Noise</th>
      <th colspan="3">Sym. 60%</th>
      <th colspan="3">Asym. 40%</th>
      <th colspan="3">Inst. 40%</th>
      <th rowspan="4">Description</th>
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
      <td rowspan="4">
        Table 2. The AUC values of Loss and Wrong Event. The experiment is conducted with ResNet18 on CIFAR100 under Sym. 60%, Asym. 40%, Inst. 40%. We use the metric AUC, which comprehensively considers accuracy, precision, and recall, to measure the selecting ability of selective metrics at different thresholds. The larger the AUC, the stronger the classification ability of the metric. It is conducted with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=128.
      </td>
    </tr>
    <tr>
      <td>Loss</td>
      <td>0.85</td>
      <td>0.69</td>
      <td>0.60</td>
      <td>0.62</td>
      <td>0.56</td>
      <td>0.55</td>
      <td>0.79</td>
      <td>0.61</td>
      <td>0.59</td>
    </tr>
    <tr>
      <td>Wrong Event</td>
      <td>0.87</td>
      <td>0.92</td>
      <td>0.91</td>
      <td>0.72</td>
      <td>0.75</td>
      <td>0.74</td>
      <td>0.86</td>
      <td>0.88</td>
      <td>0.88</td>
    </tr>
  </tbody>
</table>


### Figure 1

<div>			
    <center>	
    <img src="Reviewer-6YuD/e360fc7276717ab61b489d57cc5eada.png"
         alt=""
         style=""/>
    <br>		
    </center>
</div>




