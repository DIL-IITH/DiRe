Step 1: Squeeze : To obtain the teacher model 

```bash
sh squeeze_cifar.sh
```



Step 2: obtain embeddings

```bash
python3 -W ignore get_embeddings_cifar100.py --model_path save/cifar100/resnet18_E200
```


Step 3: Generate synthetic images. Different hyperparameters can be configured based upon the GPU capacity. Below is an example of generating IPC=100 images. 

```bash
sh generate_images.sh
```
 

Step 4: Calculate test set accuracy for a given IPC 
Example: For IPC=10 run the below command 

```bash
sh relabel_cifar.sh cuda:0 10 syn_data/cifar100_regularized dire
```

