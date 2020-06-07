# Deploying PyTorch Models to Production via Amazon SageMaker

## Machine Learning Workflow
Below diagram describes the overall workflow pattern in a typical machine learning model deployment.<br><br>
<img src="./images/machine_learning_workflow.png"></img>

## Setting up Amazon SageMaker Account

### Step 1: Setting up a notebook instance

Sign-in to your amazon-sagemaker console account and search for *Amazon SageMaker* under *Machine Learning* heading.<br><br>
<img src="./images/1. AWS console dashboard.png"></img><br><br>

After clicking on *Amazon SageMaker*, you will land on *Sagemaker's dashboard*. Here, click on *Notebook instances* under *Notebook* section on the right.<br><br>
<img src="./images/1. AWS console dashboard.png"></img><br><br>

*Notebook instances* dashboard should look something like this, if there are no instances of notebook running.<br><br>
<img src="./images/3. zero notebook instances.png"></img><br><br>

In order to create a notebook, click on *Create Notebook Instance* in the top right corner. After this, give a name to your notebook.<br><br>
<img src="./images/4. name your notebook.png"></img><br><br>


Next, under IAM role select Create a new role. You should get a pop-up window that looks like the one below. The only change that needs to be made is to select None under S3 buckets you specify, as is shown in the image below.<br><br>
<img src="./images/5. create IAM role.png"></img><br><br>

Final screenshot before running the notebook instance should look something like this -<br><br>
<img src="./images/6. notebook instance settings.png"></img><br><br>

*Note:* Your notebook name may be different than the one displayed and the IAM role that appears will be different.<br><br>

Now scroll down and click on Create notebook instance.<br><br>
<img src="./images/7. running notebook dashboard.png"></img><br><br>

Once your notebook instance has started and is accessible, click on open to get to the Jupyter notebook main page.<br><br>



