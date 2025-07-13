import torch

# Create a sample tensor
tensor = torch.randn(2, 768, 576)

# Create the index list
index_list = [[1, 4, 199, 3, 5], [1, 4, 199, 3, 5]]
output = []
# Perform index_select

for i in range(2):
    result = torch.index_select(tensor, dim=2, index=torch.tensor(index_list[i]))[i, :, :]
    output.append(result)
    # result = torch.stack((result,y), dim = 0)
stacked_tensor = torch.stack(output)

print(stacked_tensor.shape)
print(len(output))
# Print the result shape
print(result.shape)
