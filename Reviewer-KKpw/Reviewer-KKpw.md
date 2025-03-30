## 📊 Figures and Tables

### Figure 1

![e67b724c167931bb78a7946a9b4afcc](e67b724c167931bb78a7946a9b4afcc.png)

<center>Figure 1. The results of Ranks Changes Extremes of Loss and Wrong Event. The experiment is conducted under the setting with ResNet18 from scratch on CIFAR-10 with Sym. 60%, with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=64. The experiment was performed on a single A100 80GB</center>

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
      <td>5</td>
      <td>15</td>
      <td>25</td>
      <td>5</td>
      <td>15</td>
      <td>25</td>
      <td>5</td>
      <td>15</td>
      <td>25</td>
    </tr>
    <tr>
      <td>Loss</td>
      <td>0.96</td>
      <td>0.88</td>
      <td>0.74</td>
      <td>0.66</td>
      <td>0.63</td>
      <td>0.59</td>
      <td>0.84</td>
      <td>0.77</td>
      <td>0.68</td>
    </tr>
    <tr>
      <td>Wrong Event</td>
      <td>0.97</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>0.76</td>
      <td>0.80</td>
      <td>0.79</td>
      <td>0.92</td>
      <td>0.95</td>
      <td>0.94</td>
    </tr>
  </tbody>
</table>
<center>Table 1. The AUC values of Loss and Wrong Event. The experiment is conducted under the setting with pre-trained ResNet50 on CIFAR-100 under three noise settings, with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=64. The experiment was performed on a single A100 80GB, repeated 5 times</center>

### Figure 2

![image-20250329160530629](image-20250329160530629.png)

<center>Figure 2. The results of Loss, Wrong Event and Random. The experiment is conducted under the setting with ResNet-18 on CIFAR-10 with Inst. 40%, with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=64. The experiment was performed on a single A100 80GB.</center>

### Table 2

| Noise                  | Sym. 60% | Asym. 40% | Inst. 40% |
| ---------------------- | -------- | --------- | --------- |
| Single Loss + BMM      | 75.3     | 69.7      | 68.4      |
| Accumulated Loss + GMM | 79.2     | 74.5      | 75.9      |
| Accumulated Loss + BMM | 80.1     | 75.9      | 77.1      |
| Wrong event + BMM      | 80.8     | 78.0      | 82.9      |

<center>Table2. The results of single loss, acculated loss and wrong event. The experiment is conducted under the setting with ResNet-18 on CIFAR-10 with Inst. 40%, with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=64. The experiment was performed on a single A100 80GB.</center>

### Table 3

| Start Model   | Sym. 60% | Asym. 40% | Inst. 40% |
| ------------- | -------- | --------- | --------- |
| Initial Model | 80.1     | 77.1      | 83.3      |
| Base Model    | 81.3     | 77.6      | 83.8      |

<center>Table3. The results of Initial Model and Base Model. The experiment is conducted under the setting with ResNet-50 on CIFAR-100, with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=64. The experiment was performed on a single A100 80GB, repeated 5 times.</center>

### Table 4

| $$\epsilon(\cdot)$$          | Sym. 60% | Asym. 40% | Inst. 40% |
| ---------------------------- | -------- | --------- | --------- |
| Without$$\epsilon(\cdot)$$=0 | 78.2     | 70.4      | 77.3      |
| Fixed$$\epsilon(\cdot)$$=1   | 80.3     | 76.8      | 82.8      |
| Dynamic$$\epsilon(\cdot)$$   | 81.1     | 77.5      | 83.7      |

<center>Table4. The experiment is conducted under the setting with Pretrained ResNet-50 on CIFAR-100, with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=64. The experiment was performed on a single A100 80GB, repeated 5 times.</center>

### Table 5

| Noise                   | Sym. 60% | Asym. 40% | Inst. 40% |
| ----------------------- | -------- | --------- | --------- |
| Loss + Total BMM        | 79.9     | 69.9      | 74.6      |
| Loss + Class BMM        | 80.8     | 76.0      | 79.8      |
| Wrong event + Total BMM | 80.3     | 75.7      | 81.5      |
| Wrong event + Class BMM | 81.2     | 78.3      | 83.2      |

<center>Table5. The experiment is conducted under the setting with Pretrained ResNet-50 on CIFAR-100, with AdamW, lr = 1e-3, weight_decay = 1e-5, batch_size=64. The experiment was performed on a single A100 80GB, repeated 5 times.</center>