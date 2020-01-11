# Long Short Term Memory (LSTM)

## Basics of LSTM

Basic RNN was unable to retain long term memory to make prediction regarding the current picture is that od a wolf or dog. This is where LSTM comes into picture. In an LSTM, we would expect the following behaviour -


| Expected Behaviour of LSTM                                                                   | Reference Diagram                                                       |
|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| 1. Long Term Memory (LTM) and Short Term Memory (STM) to combine and produce correct output. | <img src="./images/05. lstm_basics_1.png" width="300px" height="230px"> |
| 2. LTM and STM and event should update the new LTM.                                          | </img>  <img src="./images/06. lstm_basics_2.png" width="530px" height="250px"></img>  |
| 3. LTM and STM and event should update the new STM.                                          | <img src="./images/07. lstm_basics_3.png" width="530px" height="250px"></img>          |
