# Long Short Term Memory (LSTM)

## Basics of LSTM

Basic RNN was unable to retain long term memory to make prediction regarding the current picture is that od a wolf or dog. This is where LSTM comes into picture. In an LSTM, we would expect the following behaviour -


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

<img src="./images/12.lean_gate_equation.png"></img>

## Forget Gate
Forget gate takes into account the LTM and decides which part of it to keep and which part of LTM is useless and forgets it. LTM gets multiplied by a **forget factor** inroder to forget useless parts of LTM. <br>
<img src="./images/13. forget_gate.png"></img>



