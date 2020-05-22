# Long Short Term Memory (LSTM)

## Basics of LSTM

Basic RNN was unable to retain long term memory to make prediction regarding the current picture is that od a wolf or dog. This is where LSTM comes into picture. The LSTM cell allows a recurrent system to learn over many time steps without the fear of losing information due to the vanishing gradient problem. It is fully differentiable, therefore gives us the option of easily using backpropagation when updating the weights. Below is the a sample mathematical model of an LSTM cell - <br>

<img src="./images/01.lstm_cell.png"></img><br>


In an LSTM, we would expect the following behaviour -


| Expected Behaviour of LSTM                                                                   | Reference Diagram                                                       |
|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| 1. Long Term Memory (LTM) and Short Term Memory (STM) to combine and produce correct output. | <img src="./images/05. lstm_basics_1.png" width="300px" height="230px"> |
| 2. LTM and STM and event should update the new LTM.                                          | </img>  <img src="./images/06. lstm_basics_2.png" width="530px" height="250px"></img>  |
| 3. LTM and STM and event should update the new STM.                                          | <img src="./images/07. lstm_basics_3.png" width="530px" height="250px"></img>          |



## How LSTMs work?

| LSTM consists of 4 types of gates -  <br>1. Forget Gate<br>  2. Learn Gate<br> 3. Remember Gate<br> 4. Use Gate<br> | <img src="./images/10. lstm_architecture_02.png" width="530px" height="250px"></img> |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|

### LSTM Explained
Assume the following - 
1. LTM = Elephant
2. STM = Fish
3. Event = Wolf/Dog

| LSTM Operations                                                                                                                                                                                            | Reference Video                                      |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| **LSTM places LTM, STM and Event as follows -**<br> 1. Forget Gate = LTM<br>  2. Learn Gate = STM + Event<br> 3. Remember Gate = LTM + STM + Event<br> 4. Use Gate = LTM + STM + Event<br> 5. In the end, LTM and STM are updated.<br> | <img src="./images/Animated GIF-downsized_large.gif"></img> |

## Learn Gate
Learn gate takes into account **short-term memory and event** and then ignores a part of it and retains only a part of information.<br>
<img src="./images/11. learn_gate.png" height="200px" width="500px"></img>

### Mathematically Explained
STM and Event are combined together through **activation function** (tanh), which we further multiply it by a **ignore factor** as follows -<br>

<img src="./images/12.lean_gate_equation.png" height="200px" width="500px"></img>

## Forget Gate
Forget gate takes into account the LTM and decides which part of it to keep and which part of LTM is useless and forgets it. LTM gets multiplied by a **forget factor** inroder to forget useless parts of LTM. <br>
<img src="./images/13. forget_gate.png" height="200px" width="500px"></img>

## Remember Gate
Remember gate takes LTM coming from Forget gate and STM coming from Learn gate and combines them together. Mathematically, remember gate adds LTM and STM.<br><br>
<img src="./images/14. remember_gate.png" height="200px" width="400px"></img> <img src="./images/15. remember_gate_equation.png" height="200px" width="450px"></img>

## Use Gate
Use gate takes what is useful from LTM and what's useful from STM and generates a new LTM.<br><br>
<img src="./images/16. use_gate.png" height="200px" width="400px"></img> <img src="./images/17. use_gate_equation.png" height="200px" width="450px"></img>






