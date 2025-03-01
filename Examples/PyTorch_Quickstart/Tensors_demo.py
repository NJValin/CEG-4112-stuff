#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import numpy as np


# In[4]:


data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)


# In[5]:


np_array = np.array(data)
print(np_array)
x_np = torch.from_numpy(np_array)
print(x_np)


# In[6]:


x_ones = torch.ones_like(x_data) # retain the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # Override the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# In[8]:


shape = (2, 3,) # Shape is a tuple of tensor dimensions.
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# In[9]:


device = 'cuda'


# In[10]:


tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# In[11]:


tensor = tensor.to(device)


# In[12]:


print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# In[ ]:




